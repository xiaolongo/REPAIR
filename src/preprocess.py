"""Filter graph classes and prepare public training splits and sealed labels."""
import argparse
from pathlib import Path
import numpy as np
from src.numerics import canonicalize_edges
from src.splits import make_splits
from src.io import read_json,write_json,write_npz,file_hash
from src.data_io import manifest

def prepare(raw,texts,classes,out,features=None):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    with np.load(raw,allow_pickle=False) as f:ids=f['node_ids'];x=f['x'] if 'x' in f else None;edge=f['edge_index'];labels=f['raw_labels']
    if features:
        with np.load(features,allow_pickle=False) as f:
            if not np.array_equal(ids,f['node_ids']):raise ValueError('Feature row order differs from raw graph')
            x=f['x'].copy()
    if x is None:raise ValueError('Supply x in raw graph or use --features')
    if x.shape!=(len(ids),384) or x.dtype!=np.float32 or not np.isfinite(x).all():raise ValueError('Supply Nomic-384 float32 x aligned with node_ids')
    counts={int(c):int((labels==c).sum()) for c in np.unique(labels)};kept=[c for c,n in counts.items() if n>=160]
    rows=np.flatnonzero(np.isin(labels,kept));remap=np.full(len(ids),-1,dtype=np.int64);remap[rows]=np.arange(len(rows))
    edge=remap[edge[:,np.isin(edge,rows).all(0)]];edge,_=canonicalize_edges(edge,len(rows));ids,x,labels=ids[rows],x[rows],labels[rows]
    text=read_json(texts);defs=read_json(classes);out.mkdir(parents=True)
    write_npz(out/'graph.npz',node_ids=ids,x=x,edge_index=edge.numpy());write_json(out/'node_texts.json',{n:text[n] for n in ids})
    for seed in [0,1,2]:
        roles,targets,info=make_splits(labels,dict(class_split_seed=11,num_id_classes=len(kept)//2,seeds=[100+seed],id_train_per_class=100,id_val_per_class=50,candidate_budget=2000,candidate_fraction_cap=.4,fold_seed=23))
        if len(roles['A'])+len(roles['B'])!=2000:raise ValueError('Insufficient nodes for the paper candidate budget')
        write_npz(out/f'seed{seed}.npz',targets=targets.numpy(),**roles)
    definitions=[dict(id=str(i),raw_class=int(c),name=defs[str(c)]['name'],definition=defs[str(c)].get('definition','Not provided.')) for i,c in enumerate(info['id_classes'])]
    write_json(out/'id_definitions.json',definitions);mapping={c:i for i,c in enumerate(info['id_classes'])};targets=np.array([mapping.get(int(c),-100) for c in labels])
    write_npz(out/'sealed/truth.npz',node_ids=ids,class_targets=targets,is_ood=(targets<0).astype(np.int64))
    write_json(out/'sealed/manifest.json',dict(files={'truth.npz':file_hash(out/'sealed/truth.npz')}));manifest(out,dict(kind='new preprocessing',class_seed=11,node_seeds=[100,101,102],fold_seed=23,removed_classes=[c for c in counts if c not in kept]))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True);p.add_argument('--texts',type=Path,required=True);p.add_argument('--classes',type=Path,required=True);p.add_argument('--features',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(a.raw,a.texts,a.classes,a.output,a.features)
if __name__=='__main__':main()
