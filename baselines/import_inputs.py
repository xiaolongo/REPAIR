"""Import cached LECT synthetic features or LG-Plug dynamic-codebook selections."""
import argparse
from pathlib import Path
import numpy as np
from src.data import load
from src.io import read_json,write_json,write_npz,file_hash

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--method',choices=['lect','lgplug'],required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--source',type=Path,required=True);a=p.parse_args()
    view,_,_=load(a.data,a.seed);out=a.data/a.method/f'seed{a.seed}'
    if out.exists():raise FileExistsError(out)
    if a.method=='lect':
        nodes=read_json(a.source/'nodes.json')
        with np.load(a.source/'features.npz',allow_pickle=False) as f:x=f['x'].copy()
        if x.shape!=(len(nodes),384) or not np.isfinite(x).all():raise ValueError('Invalid synthetic features')
        for n in nodes:
            if not 1<=len(n['neighbors'])<=6 or not set(n['neighbors'])<=set(view.roles['L']):raise ValueError('Synthetic edges must connect to L')
        write_json(out/'nodes.json',nodes);write_npz(out/'features.npz',x=x);files=['nodes.json','features.npz']
    else:
        with np.load(a.source/'exposure.npz',allow_pickle=False) as f:rows=f['rows'].astype(np.int64)
        if len(np.unique(rows))!=len(rows) or not set(rows)<=set(np.r_[view.roles['A'],view.roles['B']]):raise ValueError('LG exposure must be within C')
        write_npz(out/'exposure.npz',rows=rows,node_ids=np.asarray(view.graph.node_ids)[rows]);files=['exposure.npz']
    write_json(out/'complete.json',dict(passed=True,source_sha256={n:file_hash(a.source/n) for n in files},files={n:file_hash(out/n) for n in files}))
if __name__=='__main__':main()
