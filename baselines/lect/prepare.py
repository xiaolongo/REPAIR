"""Generate synthetic nodes locally, then encode their original generated text."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from src.io import ROOT,read_json,write_json,write_npz,lock_run,seal,file_hash
from src.data import load
from src.local_llm import LocalJSON
from src.encode import run as encode
from baselines.lect.model import generation_plan,validate_sample

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--endpoint',required=True);p.add_argument('--model',default='Qwen3.8-27B');p.add_argument('--nomic-model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    view,_,_=load(a.data,a.seed);cfg=read_json(ROOT/'configs/lect.json');out=a.output.resolve()
    lock_run(out,dict(input_sha256=file_hash(a.data/'manifest.json'),seed=a.seed,model=a.model,endpoint=a.endpoint))
    client=LocalJSON(a.endpoint,a.model,out/'cache',4096);nodes=generation_plan(view.roles['L'],len(view.graph.node_ids),a.seed,cfg)
    system=(ROOT/'baselines/lect/prompts/lect_system.txt').read_text().strip();template=(ROOT/'baselines/lect/prompts/lect_user.txt').read_text();defs=view.id_definitions
    categories='\n'.join(f"{d['id']}. {d['name']}: {d['definition']}" for d in defs)
    def ask(node):
        neighbors='\n'.join(defs[int(view.cls_targets[row])]['name'] for row in node['neighbors'])
        user=template.format(regime=node['regime'],node_id=node['node_id'],categories=categories,neighbors=neighbors)
        body=dict(model=a.model,messages=[dict(role='system',content=system),dict(role='user',content=user)],temperature=0,max_tokens=768,chat_template_kwargs={'enable_thinking':False},response_format={'type':'json_object'},seed=a.seed)
        return dict(node=node,**client.ask_json(body,validate_sample,'Return exactly the five requested nonempty JSON string fields, without code fences. Complete the original task.'))
    with ThreadPoolExecutor(max_workers=8) as pool:responses=list(pool.map(ask,nodes))
    valid=[r for r in responses if r['status']=='valid']
    if not valid:raise ValueError('No valid synthetic node; training cannot continue')
    write_json(out/'responses.json',responses);write_json(out/'nodes.json',[r['node'] for r in valid])
    ids=[r['node']['node_id'] for r in valid];write_npz(out/'node_order.npz',node_ids=ids)
    write_json(out/'texts.json',{r['node']['node_id']:r['value']['title']+'\n'+r['value']['description'] for r in valid})
    encode(out/'node_order.npz',out/'texts.json',a.nomic_model,out/'features.npz',reject_long=True)
    seal(out,['run.json','nodes.json','features.npz','responses.json'])
if __name__=='__main__':main()
