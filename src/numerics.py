"""Graph operators, GCN backbones and evaluation metrics."""
import os, random, hashlib, json
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score


def json_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      default=lambda x: str(x) if isinstance(x, os.PathLike) else x.tolist(),
                      indent=2).encode("utf-8")

def digest(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()

def array_hash(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(json_bytes([str(a.dtype), list(a.shape)]))
    h.update(a.tobytes())
    return h.hexdigest()

@dataclass(slots=True)
class LogitsBundle:
    logits: torch.Tensor
    hidden: torch.Tensor | None

@dataclass(slots=True)
class Propagation:
    indices: torch.Tensor
    values: torch.Tensor
    sparse: torch.Tensor
    num_nodes: int
    offdiag: torch.Tensor
    base_hash: str

@dataclass(slots=True)
class GraphData:
    node_ids: list[str]
    x: torch.Tensor
    edge_index: torch.Tensor
    feature_meta: dict
    graph_hash: str

def int_array(value, name, ndim=None):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    a = np.asarray(value)
    if a.dtype.kind not in "iu" or (ndim is not None and a.ndim != ndim):
        raise ValueError(f"{name} must be a non-bool integer array with ndim={ndim}")
    if a.size and a.max() > np.iinfo(np.int64).max:
        raise ValueError(f"{name} exceeds int64")
    return a.astype(np.int64, copy=True)

def canonicalize_edges(edge_index, num_nodes):
    edge = int_array(edge_index, "edge_index", 2)
    if edge.shape[0] != 2 or num_nodes < 1 or (edge.size and (edge.min() < 0 or edge.max() >= num_nodes)):
        raise ValueError("Invalid edge endpoints/shape")
    mask = edge[0] != edge[1]
    clean = edge[:, mask]
    pairs = np.concatenate([clean.T, clean[::-1].T])
    unique = np.unique(pairs, axis=0)
    order = np.lexsort((unique[:, 0], unique[:, 1]))
    canonical = unique[order].T.copy()
    counts = {"num_nodes": num_nodes, "input_edges": edge.shape[1], "input_self_loops": int((~mask).sum()),
              "input_duplicate_directed_edges": edge.shape[1] - len(np.unique(edge.T, axis=0)),
              "added_reverse_edges": len(unique) - len(np.unique(clean.T, axis=0)),
              "output_directed_edges": len(unique), "output_undirected_edges": len(unique) // 2,
              "isolated_nodes": num_nodes - len(np.unique(canonical))}
    return torch.from_numpy(canonical), counts

def build_normalized_operator(edge_index, num_nodes, dtype=torch.float32):
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("Propagation requires float32 or float64")
    edge = edge_index.detach().cpu()
    canonical, _ = canonicalize_edges(edge, num_nodes)
    if edge.dtype != torch.int64 or not torch.equal(canonical, edge):
        raise ValueError("Input must be a canonical loop-free undirected simple graph")
    loops = torch.arange(num_nodes)
    src = torch.cat((edge[0], loops))
    dst = torch.cat((edge[1], loops))
    degree = torch.bincount(dst, minlength=num_nodes).to(dtype)
    weights = (degree[dst] * degree[src]).rsqrt()
    with torch.sparse.check_sparse_tensor_invariants():
        sparse = torch.sparse_coo_tensor(torch.stack((dst, src)), weights, (num_nodes, num_nodes),
                                         check_invariants=True).coalesce()
    indices, values = sparse.indices(), sparse.values()
    sha = digest({"indices": array_hash(indices.numpy()), "values": array_hash(values.numpy()),
                  "num_nodes": num_nodes, "formula": "D^-1/2 (A+I) D^-1/2", "rows": "receivers"})
    return Propagation(indices, values, sparse, num_nodes, indices[0] != indices[1], sha)

def make_classifier(input_dim, hidden_dim, num_classes, seed):
    if min(input_dim, hidden_dim) < 1 or num_classes < 2:
        raise ValueError("Invalid classifier dimensions")
    # CPU construction uses Linear.reset_parameters defaults and leaves caller RNG intact.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = torch.nn.ModuleDict({"layer0": torch.nn.Linear(input_dim, hidden_dim),
                                     "layer1": torch.nn.Linear(hidden_dim, num_classes)})
    model.structure = {"input_dim": input_dim, "hidden_dim": hidden_dim, "num_classes": num_classes,
                       "seed": seed, "initialization": "torch.nn.Linear.reset_parameters defaults"}
    return model

def classifier_forward(model, x, kind, s_first, s_second, dropout=0.2):
    if not 0 <= dropout < 1 or kind not in ("gcn", "mlp"):
        raise ValueError("Invalid classifier kind/dropout")
    l0, l1 = model["layer0"], model["layer1"]
    if kind == "gcn":
        for s in (s_first, s_second):
            if s is None or s.layout != torch.sparse_coo or s.shape != (len(x), len(x)):
                raise ValueError("GCN requires two explicit COO propagation matrices")
        h = F.relu(torch.sparse.mm(s_first, F.linear(x, l0.weight)) + l0.bias)
        dropped = F.dropout(h, p=dropout, training=model.training)
        z = torch.sparse.mm(s_second, F.linear(dropped, l1.weight)) + l1.bias
    else:
        h = F.relu(l0(x))
        z = l1(F.dropout(h, p=dropout, training=model.training))
    return LogitsBundle(z, h)

def parameter_hash(model):
    return digest({k: array_hash(v.detach().cpu().contiguous().numpy()) for k, v in sorted(model.state_dict().items())})

def freeze_backbone(model):
    model.eval()
    model.requires_grad_(False)
    for p in model.parameters():
        p.grad = None
    return parameter_hash(model)

def seed_everything(seed: int, deterministic: bool = True) -> dict:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    return {"seed": seed, "deterministic": deterministic, "cross_platform_bitwise": False}

def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu(v) for v in value)
    return value

def _rng_state():
    state = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
            "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]]}

def _restore_rng(state):
    random.setstate(state["python"])
    name, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])

def propagate(score, edge_index, steps=2, alpha=0.5):
    if score.ndim != 1 or steps < 0 or not 0 <= alpha <= 1:
        raise ValueError('Expected a score vector, nonnegative steps and alpha in [0,1]')
    row, col = edge_index.to(score.device)
    degree = torch.bincount(col, minlength=len(score)).float()
    value = torch.nan_to_num(degree[col].reciprocal(), nan=0., posinf=0., neginf=0.)
    adj = torch.sparse_coo_tensor(torch.stack((col, row)), value.to(score.dtype),
                                  (len(score), len(score)), device=score.device).coalesce()
    e = score[:, None]
    for _ in range(steps):
        e = alpha*e + (1-alpha)*torch.sparse.mm(adj, e)
    return e[:, 0]

def detection_metrics(ood_scores,is_ood,recall_level=.95):
    ood_scores=np.asarray(ood_scores);is_ood=np.asarray(is_ood)
    if ood_scores.ndim!=1 or is_ood.shape!=ood_scores.shape or not np.isfinite(ood_scores).all():
        raise ValueError('Finite aligned score and label vectors required')
    if not np.array_equal(np.unique(is_ood),[0,1]) or not 0<recall_level<=1:
        raise ValueError('Both ID and OOD and a valid recall level are required')
    y=is_ood==0
    score=-ood_scores
    order=np.argsort(score,kind='mergesort')[::-1]
    sorted_score=score[order];sorted_y=y[order]
    ends=np.r_[np.where(np.diff(sorted_score))[0],len(y)-1]
    tp=np.cumsum(sorted_y,dtype=np.float64)[ends]
    fp=1+ends-tp
    recall=tp/tp[-1]
    last=tp.searchsorted(tp[-1])
    recall=np.r_[recall[last::-1],1]
    fp=np.r_[fp[last::-1],0]
    cutoff=np.argmin(np.abs(recall-recall_level))
    return {'AUROC':float(roc_auc_score(y,score)),
            'AUPR':float(average_precision_score(y,score)),
            'FPR95':float(fp[cutoff]/np.sum(~y)),
            'achieved_ID_TPR':float(recall[cutoff]),
            'n_ID':int(y.sum()),'n_OOD':int((~y).sum())}

@dataclass
class IDStop:
    best: float
    anchor: float
    best_epoch: int
    last_improvement: int
    patience: int = 300
    min_delta: float = .0001

    def update(self, epoch, ce):
        selected = ce < self.best
        if selected:
            self.best, self.best_epoch = ce, epoch
        if self.anchor - ce >= self.min_delta:
            self.anchor, self.last_improvement = ce, epoch
        return selected, epoch - self.last_improvement >= self.patience
