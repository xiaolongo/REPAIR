from src.io import digest,file_hash,read_json,write_json
from baselines.lgplug.codebook import Codebook,consensus
VERSION = 'lgplug_qwen_dynamic_v1'


def request(node_id, text, book, definitions, cfg, system, template):
    messages = [{'role':'system','content':system},
                {'role':'user','content':template.format(categories='\n'.join(book.names), text=text)}]
    body = {k:cfg[k] for k in ('model','reasoning_effort','temperature','max_completion_tokens','response_format')}
    body.update(messages=messages, stream=False)
    identity = {'node_id':node_id, 'text_hash':digest(text), 'codebook':book.state(),
                'ID_definition_hash':digest(definitions), 'body_hash':digest(body),
                'prompt_version':VERSION, 'provider':cfg['provider'], 'base_url':cfg['base_url']}
    return {'node_id':node_id, 'identity':identity, 'cache_key':digest(identity), 'body':body}

def run_cluster(group, ids, texts, definitions, cfg, client, folder):
    complete_path = folder / "complete.json"
    if complete_path.exists():
        saved = read_json(complete_path)
        assert saved["group_hash"] == digest(group) and saved["config_hash"] == digest(cfg)
        for path, sha in saved["files"].items():
            assert file_hash(path) == sha
        return saved
    book = Codebook([item["name"] for item in definitions])
    records, files, stop_reason = [], [], "candidate_exhausted"
    planned = sum(group["batches"], [])
    assert len(planned) == len(set(planned)) and set(planned) <= set(group["representatives"])
    for batch_index, batch in enumerate(group["batches"]):
        current = []
        for row in batch:
            before = book.state()
            req = request(ids[row], texts[ids[row]], book, definitions, cfg, client.system, client.template)
            answer = client.ask(req)
            category = book.add(answer["label"]) if answer["status"] == "valid" else None
            event = {"row": row, "node_id": ids[row], "batch": batch_index, "status": answer["status"],
                     "label": answer["label"], "category": category, "cache_key": req["cache_key"],
                     "before": before, "after": book.state(), "attempt_ids": answer["attempt_ids"]}
            path = folder / f"node_{len(records):02}.json"
            write_json(path, event)
            files.append(path)
            records.append(event)
            current.append(event)
        if consensus(current):
            stop_reason = "batch_consensus"
            break
        if batch_index + 1 == cfg["trials"]:
            stop_reason = "max_trials"
    exposure = book.exposure(records, cfg["top_k"])
    result = {"cluster": group["cluster"], "group_hash": digest(group), "config_hash": digest(cfg),
              "book": book.state(), "top_categories": book.top(cfg["top_k"]), "exposure_rows": exposure,
              "queried_rows": [record["row"] for record in records], "logical_queries": len(records),
              "valid": sum(record["status"] == "valid" for record in records),
              "cache_keys": [record["cache_key"] for record in records], "stop_reason": stop_reason,
              "files": {str(path): file_hash(path) for path in files}}
    write_json(complete_path, result)
    return result
