"""Recompute REPAIR predictions from the final checkpoint and fitted head."""
import argparse
from pathlib import Path
import torch
from src.data import load,model,DATASETS
from src.io import read_json,verify,write_npz
from src.numerics import propagate

def predict(data,slug,seed,run,out):
    verify(read_json(run/'complete.json'));view,s,_=load(data,seed);net,forward=model(slug,view,seed)
    net.load_state_dict(torch.load(run/'final/best.pt',map_location='cpu',weights_only=True)['model']);net.eval()
    h=torch.load(run/'final/post_head/head.pt',map_location='cpu',weights_only=True);w=h['model'];norm=h['norm']
    with torch.no_grad():
        z=forward(net,view.graph.x,'gcn',s,s,.5).logits;e=-torch.logsumexp(z,1)
        raw=w['alpha']*((e.double()-norm['mean'])/norm['std'])+view.graph.x.double()@w['w']+w['b']
        score=propagate(raw,view.graph.edge_index,3,.5)
    write_npz(out,node_ids=view.graph.node_ids,logits=z.numpy(),score=score.numpy())

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--dataset',choices=DATASETS,required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();predict(a.data,a.dataset,a.seed,a.run,a.output)
if __name__=='__main__':main()
