import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from src.io import ROOT,read_json,write_json,write_npz,lock_run,seal,file_hash
from src.data import load,DATASETS
from src.local_llm import LocalJSON
from baselines.lgplug.codebook import candidate_clusters,validate_label
from baselines.lgplug.query_core import run_cluster

DOMAINS=dict(wikics='Wikipedia articles about computer science',cora_full='scientific papers',books_history='history book descriptions',ele_photo='photography product descriptions',ele_computers='computer product descriptions',arxiv='scientific papers about computer science',sportsfit='sports and fitness product titles')

def main():
    p=argparse.ArgumentParser(description='LG-Plug dynamic-codebook queries against an already running local Qwen server')
    p.add_argument('--data',type=Path,required=True);p.add_argument('--dataset',choices=DATASETS,required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--alignment',type=Path,required=True);p.add_argument('--endpoint',required=True);p.add_argument('--model',default='Qwen3.8-27B');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    view,_,_=load(a.data,a.seed);cfg=read_json(ROOT/'configs/lgplug_query.json');cfg.update(base_url=a.endpoint,model=a.model,provider='local_Qwen_Q8')
    out=a.output.resolve();lock_run(out,dict(input_sha256=file_hash(a.data/'manifest.json'),alignment_sha256=file_hash(a.alignment/'embeddings.npz'),config=cfg,seed=a.seed))
    with np.load(a.alignment/'embeddings.npz',allow_pickle=False) as f:
        if f['node_ids'].tolist()!=view.graph.node_ids:raise ValueError('Alignment node order differs')
        z=f['graph_z'].copy()
    groups=candidate_clusters(np.r_[view.roles['A'],view.roles['B']],z,view.graph.node_ids,a.seed,cfg)
    client=LocalJSON(a.endpoint,a.model,out/'cache',16384)
    client.system=(ROOT/'baselines/lgplug/prompts/lgplug_system.txt').read_text().strip().replace('Wikipedia articles about computer science',DOMAINS[a.dataset])
    client.template=(ROOT/'baselines/lgplug/prompts/lgplug_user.txt').read_text()
    def ask(req):
        body={k:v for k,v in req['body'].items() if k not in ['reasoning_effort','max_completion_tokens']};body.update(max_tokens=1000,chat_template_kwargs={'enable_thinking':False})
        r=client.ask_json(body,lambda x:validate_label(x,128),'Return only a JSON object with exactly one string field label. Select an exact listed category or a concrete new category of at most 128 characters. No explanations, no generic OOD or unknown label.',cache_key=req['cache_key'])
        return dict(status=r['status'],label=r['value'],attempt_ids=r['attempt_ids'])
    client.ask=ask;texts=read_json(a.data/'node_texts.json')
    def cluster(g):return run_cluster(g,view.graph.node_ids,texts,view.id_definitions,cfg,client,out/f"cluster{g['cluster']}")
    with ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(cluster,groups))
    rows=np.asarray(sorted(row for r in results for row in r['exposure_rows']),dtype=np.int64)
    write_npz(out/'exposure.npz',rows=rows,node_ids=np.asarray(view.graph.node_ids)[rows]);write_json(out/'summary.json',dict(exposure_count=len(rows),status='available' if len(rows) else 'N/A_empty_exposure',Q_queried=False))
    seal(out,['run.json','exposure.npz','summary.json'])
if __name__=='__main__':main()
