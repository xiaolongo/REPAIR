"""Full REPAIR only. Two cross-fold teachers, eight reviews, one final detector."""
import argparse,os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import numpy as np
import torch
from src.data import DATASETS,load,identity
from src.io import ROOT,read_json,write_json,write_npz,lock_run,verify,seal,public_inputs_only,file_hash
from src.numerics import propagate
from src.review import partition
from src.pretrain import train as pretrain

def run(data,slug,seed,out,m0=None,stage_epochs=None):
    out=Path(out).resolve();cfg=read_json(ROOT/'configs/repair.json')
    if stage_epochs is not None:cfg['GNN_update']['max_epochs']=stage_epochs
    if not torch.cuda.is_available():raise RuntimeError('The preserved REPAIR stage implementation requires CUDA')
    torch.set_num_threads(4)
    if m0 is None:m0=pretrain(data,slug,seed,out/'M0',max_epochs=stage_epochs)
    m0=Path(m0).resolve()
    lock_run(out,identity(data,slug,seed,'REPAIR',config=cfg,m0_sha256=file_hash(m0)))
    if (out/'complete.json').exists():verify(read_json(out/'complete.json'));return
    if slug=='arxiv':from src.stage_arxiv import train_stage
    elif slug=='cora_full':from src.stage_cora_full import train_stage
    else:from src.stage_plain import train_stage
    from src.data import model
    with public_inputs_only():
        view,s,labels=load(data,seed,True);net,forward=model(slug,view,seed)
        net.load_state_dict(torch.load(m0,map_location='cpu',weights_only=True)['model']);net.eval()
        with torch.no_grad():e0=-torch.logsumexp(forward(net,view.graph.x,'gcn',s,s,.5).logits,1).numpy()
        pools={f:partition(view.roles[f],labels,e0,view.roles['V'],False,.8) for f in ['A','B']}
        starts={f:m0 for f in pools};write_json(out/'pools_0.json',pools)
        for t in range(1,cfg['reviews']+1):
            if any(not pools[f]['OOD'] for f in pools):raise ValueError('Empty REPAIR exposure fold. No fallback or label substitution.')
            for f in ['A','B']:
                folder=out/f'round{t}'/f
                train_stage(view,s,starts[f],pools[f]['OOD'],cfg['head_fit'],cfg['GNN_update'],folder,seed,f)
                starts[f]=folder/'best.pt'
            updated={}
            for f,other in [('A','B'),('B','A')]:
                with np.load(out/f'round{t}'/other/'scores.npz',allow_pickle=False) as a:energy=a['energy'].copy()
                updated[f]=partition(view.roles[f],labels,energy,view.roles['V'],True,.8)
            pools=updated;write_json(out/f'pools_{t}.json',pools)
        selected=pools['A']['OOD']+pools['B']['OOD']
        train_stage(view,s,m0,selected,cfg['head_fit'],cfg['GNN_update'],out/'final',seed)
        with np.load(out/'final/scores.npz',allow_pickle=False) as a:
            score=propagate(torch.from_numpy(a['score'].copy()),view.graph.edge_index,3,.5).numpy()
            write_npz(out/'scores.npz',node_ids=view.graph.node_ids,logits=a['logits'],score=score)
        write_json(out/'exposure.json',dict(rows=selected));seal(out,['run.json','scores.npz','final/best.pt','final/post_head/head.pt','exposure.json'])

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',choices=DATASETS,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--seed',type=int,choices=[0,1,2],default=0);p.add_argument('--output',type=Path,required=True);p.add_argument('--m0',type=Path)
    a=p.parse_args();run(a.data,a.dataset,a.seed,a.output,a.m0)
if __name__=='__main__':main()
