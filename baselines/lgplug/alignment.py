import math,importlib.metadata
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from transformers import CLIPTokenizerFast
from src.io import digest,file_hash,read_json,write_json,write_npz,verify,public_inputs_only,_save_checkpoint,write_csv
from src.numerics import parameter_hash,IDStop,seed_everything,_cpu,_rng_state,_restore_rng
from baselines.lgplug.loss import alignment_loss
from baselines.lgplug.model import GraphText,alignment_split,pack_rows


def token_cache(view, public, cfg, out):
    folder = ROOT / cfg["tokenizer_path"]
    source = read_json(folder / "source_manifest.json")
    assert source["revision"] == cfg["tokenizer_revision"]
    for name, sha in source["files"].items():
        assert file_hash(folder / name) == sha
    tokenizer = CLIPTokenizerFast.from_pretrained(folder, local_files_only=True)
    tokenizer.model_max_length = cfg["max_length"]
    assert len(tokenizer) == cfg["vocab_size"]
    identity = {"ordered_ids_hash": digest(view.graph.node_ids), "texts_hash": file_hash(public / "node_texts.json"),
                "tokenizer_source_hash": file_hash(folder / "source_manifest.json"), "max_length": cfg["max_length"]}
    meta_path, cache_path = out / "tokens_meta.json", out / "tokens.npz"
    if meta_path.exists():
        meta = read_json(meta_path)
        assert meta["identity"] == identity and file_hash(cache_path) == meta["file_hash"]
    else:
        texts = read_json(public / "node_texts.json")
        tokens, masks = [], []
        for start in range(0, len(view.graph.node_ids), 256):
            batch = [texts[node] for node in view.graph.node_ids[start:start + 256]]
            values = tokenizer(batch, truncation=True, padding="max_length", max_length=cfg["max_length"], return_tensors="np")
            tokens.append(values["input_ids"].astype(np.int32))
            masks.append(values["attention_mask"].astype(np.uint8))
        write_npz(cache_path, tokens=np.concatenate(tokens), mask=np.concatenate(masks))
        write_json(meta_path, {"identity": identity, "file_hash": file_hash(cache_path), "pretrained_weights_loaded": False})
    with np.load(cache_path, allow_pickle=False) as saved:
        tokens = torch.from_numpy(saved["tokens"].astype(np.int64))
        mask = torch.from_numpy(saved["mask"].astype(np.int64))
    assert tokens.shape == (len(view.graph.node_ids), cfg["max_length"])
    special = {"bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id}
    return tokens, mask, special

def batch_loss(model, x, edges, tokens, mask, nodes, pairs, weight):
    rows, node_pos, edge_pos = pack_rows(nodes, pairs)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z = model.graph(x, edges)
        h = model.texts(tokens[rows].cuda(), mask[rows].cuda())
    return alignment_loss(z[nodes].float(), h[node_pos].float(), z[pairs].float(), h[edge_pos].float(), model.scale(), weight)

def validation(model, x, edges, tokens, mask, split, cfg):
    model.eval()
    val, pairs = split["validation_nodes"], split["validation_edges"]
    lookup = np.full(len(x), -1, dtype=np.int64)
    lookup[val] = np.arange(len(val))
    assert np.all(lookup[pairs] >= 0)
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model.graph(x, edges).float()
            h = torch.cat([model.texts(tokens[rows].cuda(), mask[rows].cuda()).float()
                           for rows in np.array_split(val, math.ceil(len(val) / 64))])
        losses = []
        for start in range(0, len(val), cfg["node_batch"]):
            rows = val[start:start + cfg["node_batch"]]
            parts = alignment_loss(z[rows], h[start:start + len(rows)], z[pairs], h[lookup[pairs]], model.scale(), cfg["edge_weight"])
            losses.append(float(parts["node"]))
        node, edge = float(np.mean(losses)), float(parts["edge"])
    return {"node": node, "edge": edge, "total": node + cfg["edge_weight"] * edge}

def run_seed(seed, cfg, out, common, smoke=False):
    with public_inputs_only():
        torch.set_num_threads(4)
        if not smoke and (out / "complete.json").exists():
            verify(read_json(out / "complete.json"))
            return
        public = Path(cfg["data_source"])
        view = load_training_view(public / "manifest.json")
        tokens, mask, special = token_cache(view, public, cfg, common)
        split = alignment_split(len(tokens), view.graph.edge_index.numpy(), cfg["split_seed_offset"] + seed,
                                cfg["validation_nodes"], cfg["validation_edges"])
        assert len(split["validation_nodes"]) % cfg["node_batch"] == 0
        seed_everything(seed, deterministic=True)
        x, edges = view.graph.x.cuda(), view.graph.edge_index.cuda()
        model = GraphText(x.shape[1], cfg, special).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        rng = np.random.default_rng(seed)
        sources = [Path(__file__), ROOT / "configs/lgplug_alignment.json", ROOT / "baselines/lgplug/loss.py",
                   ROOT / "baselines/lgplug/model.py", ROOT / "src/numerics.py",
                   common / "tokens_meta.json", public / "manifest.json"]
        protocol = {"seed": seed, "config": cfg, "token_ids": special, "graph_hash": view.graph.graph_hash,
                    "sources": {str(p): file_hash(p) for p in sources}, "labels_used": False,
                    "runtime": {name: importlib.metadata.version(name) for name in ("torch", "torch-geometric", "transformers", "numpy")},
                    "parameters": sum(p.numel() for p in model.parameters()), "pretrained_model_weights": False,
                    "train_node_count": len(split["train_nodes"]), "train_edge_count": len(split["train_edges"]),
                    "validation_node_count": len(split["validation_nodes"]), "validation_edge_count": len(split["validation_edges"])}
        identity = digest(protocol)
        history, val_history, stop, best, first_step, previous_seconds = [], [], None, None, 1, 0.
        progress_path = out / "progress.pt"
        if not smoke:
            write_json(out / "protocol.json", protocol)
            write_npz(out / "split.npz", **split)
            if progress_path.exists():
                saved = torch.load(progress_path, weights_only=True, map_location="cpu")
                assert saved["identity"] == identity
                model.load_state_dict(saved["model"])
                optimizer.load_state_dict(saved["optimizer"])
                _restore_rng(saved["rng"])
                rng.bit_generator.state = saved["sampler_state"]
                history, val_history, best = saved["history"], saved["validation_history"], saved["best"]
                stop = IDStop(**saved["stop"]) if saved["stop"] is not None else None
                first_step, previous_seconds = saved["step"] + 1, saved["seconds"]
                print({"seed": seed, "resumed_at_step": first_step}, flush=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = perf_counter()
        max_step = 2 if smoke else cfg["max_step"]
        step = first_step - 1
        done = bool(stop and step - stop.last_improvement >= stop.patience)
        for step in (() if done else range(first_step, max_step + 1)):
            nodes = rng.choice(split["train_nodes"], cfg["node_batch"], replace=False)
            pairs = split["train_edges"][rng.choice(len(split["train_edges"]), cfg["edge_batch"], replace=False)]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            parts = batch_loss(model, x, edges, tokens, mask, nodes, pairs, cfg["edge_weight"])
            assert all(torch.isfinite(v) for v in parts.values())
            parts["total"].backward()
            gradients = [p.grad.detach().norm() for p in model.parameters() if p.grad is not None]
            norm = torch.stack(gradients).norm()
            assert torch.isfinite(norm)
            optimizer.step()
            row = {"step": step, **{k: float(v.detach()) for k, v in parts.items()}, "gradient_L2": float(norm),
                   "scale": float(model.scale().detach().exp())}
            history.append(row)
            if step == 1 or step % cfg["validation_interval"] == 0 or smoke:
                print({"seed": seed, "train": row}, flush=True)
            if smoke:
                continue
            if step % cfg["validation_interval"] == 0:
                val = validation(model, x, edges, tokens, mask, split, cfg)
                assert np.isfinite(list(val.values())).all()
                selected = False
                if step == cfg["min_step"]:
                    stop = IDStop(val["total"], val["total"], step, step,
                                  cfg["patience_checks"] * cfg["validation_interval"], cfg["min_delta"])
                    selected = True
                elif step > cfg["min_step"]:
                    selected, done = stop.update(step, val["total"])
                if selected:
                    best = {"step": step, "model": _cpu(model.state_dict()), "validation": val}
                check = {"step": step, **val, "best_step": stop.best_epoch if stop else 0,
                         "last_significant_improvement": stop.last_improvement if stop else 0}
                val_history.append(check)
                print({"seed": seed, "validation": check}, flush=True)
            if step % 1000 == 0 or done:
                torch.cuda.synchronize()
                _save_checkpoint(progress_path, {"identity": identity, "step": step, "model": _cpu(model.state_dict()),
                                 "optimizer": _cpu(optimizer.state_dict()), "rng": _rng_state(), "sampler_state": rng.bit_generator.state,
                                 "history": history, "validation_history": val_history, "best": best,
                                 "stop": vars(stop) if stop else None,
                                 "seconds": previous_seconds + perf_counter() - started}, progress=True)
            if done:
                break
        torch.cuda.synchronize()
        seconds = previous_seconds + perf_counter() - started
        if smoke:
            val = validation(model, x, edges, tokens, mask, split, cfg)
            result = {"synthetic": False, "engineering_only": True, "steps": 2, "class_labels_used": False,
                      "parameters": protocol["parameters"], "history": history, "validation": val,
                      "peak_GiB": torch.cuda.max_memory_allocated() / 2**30, "train_seconds": seconds}
            write_json(common / "engineering_smoke.json", result)
            print(result, flush=True)
            return
        assert best is not None and stop is not None
        _save_checkpoint(out / "last.pt", {"step": step, "model": _cpu(model.state_dict()),
                         "optimizer": _cpu(optimizer.state_dict()), "rng": _rng_state(), "sampler_state": rng.bit_generator.state})
        _save_checkpoint(out / "selected.pt", best)
        model.load_state_dict(torch.load(out / "selected.pt", weights_only=True, map_location="cpu")["model"])
        confirmed = validation(model, x, edges, tokens, mask, split, cfg)
        assert all(abs(confirmed[k] - best["validation"][k]) < 1e-6 for k in confirmed)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            z = model.graph(x, edges).float().cpu().numpy()
        assert np.isfinite(z).all()
        write_npz(out / "embeddings.npz", node_ids=np.asarray(view.graph.node_ids), graph_z=z)
        write_csv(out / "history.csv", history)
        write_csv(out / "validation.csv", val_history)
        files = [out / name for name in ("protocol.json", "split.npz", "selected.pt", "last.pt", "embeddings.npz", "history.csv", "validation.csv")]
        record = {"seed": seed, "stop_step": step, "selected_step": best["step"], "stop_reason": "validation_patience" if done else "max_step",
                  "validation": confirmed, "parameter_hash": parameter_hash(model), "train_seconds": seconds,
                  "peak_GiB": torch.cuda.max_memory_allocated() / 2**30, "labels_used": False, "LLM_calls": 0,
                  "files": {str(p): file_hash(p) for p in files}}
        write_json(out / "complete.json", record)
        print({k: v for k, v in record.items() if k != "files"}, flush=True)
