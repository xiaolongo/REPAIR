from pathlib import Path
import numpy as np
import torch
from src.io import read_json,verify,file_hash,ROOT
from src.numerics import GraphData,digest,build_normalized_operator,canonicalize_edges
from src.splits import build_training_view

DATASETS=tuple(read_json(ROOT/'configs/datasets.json'))

def load(data,seed,require_annotations=False):
    data=Path(data).resolve()
    meta=read_json(data/'manifest.json')
    # The public manifest deliberately excludes sealed files.
    verify(meta['files'],data)
    with np.load(data/'graph.npz',allow_pickle=False) as f:
        ids=f['node_ids'].tolist();x=torch.from_numpy(f['x'].copy());edge=torch.from_numpy(f['edge_index'].copy())
    if x.dtype!=torch.float32 or x.ndim!=2 or x.shape!=(len(ids),384) or not torch.isfinite(x).all():raise ValueError('Expected finite N x 384 float32 features')
    if len(set(ids))!=len(ids):raise ValueError('Duplicate node IDs')
    canonical,_=canonicalize_edges(edge,len(ids))
    if not torch.equal(edge,canonical):raise ValueError('Graph must be symmetric, loop-free and canonical')
    with np.load(data/f'seed{seed}.npz',allow_pickle=False) as f:
        y=torch.from_numpy(f['targets'].copy());roles={k:f[k].copy() for k in ['L','V','A','B','Q']}
    defs=read_json(data/'id_definitions.json')
    view=build_training_view(GraphData(ids,x,edge,{},file_hash(data/'graph.npz')),roles,y,defs)
    labels=np.full(len(ids),-1,dtype=np.int64)
    path=data/f'annotations/seed{seed}.json'
    if path.exists():
        records=read_json(path);lookup={n:i for i,n in enumerate(ids)};candidate=set(np.r_[roles['A'],roles['B']].tolist())
        if len(records)!=len(candidate) or {lookup[r['node_id']] for r in records}!=candidate:raise ValueError('Annotations must cover exactly A union B')
        for r in records:
            if r['status']=='valid':
                decision=r['annotation']['decision']
                if decision not in ['ID','IND','OOD']:raise ValueError('Invalid annotation decision')
                labels[lookup[r['node_id']]]=int(decision=='OOD')
    elif require_annotations:raise FileNotFoundError(f'Import fixed annotations first: {path}')
    return view,build_normalized_operator(edge,len(ids)).sparse,labels

def model(slug,view,seed):
    from src import numerics
    if slug=='arxiv':from src import model_arxiv as api
    elif slug=='cora_full':from src import model_cora_full as api
    else:api=numerics
    cfg=read_json(ROOT/'configs/datasets.json')[slug]
    return api.make_classifier(384,cfg['hidden'],len(view.id_definitions),seed),api.classifier_forward

def identity(data,slug,seed,method,**extra):
    return dict(dataset=slug,seed=seed,method=method,input_sha256=file_hash(Path(data)/'manifest.json'),**extra)
