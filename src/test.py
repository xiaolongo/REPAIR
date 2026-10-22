"""Evaluate locked predictions. This is the only CLI that reads sealed truth."""
import argparse
from pathlib import Path
import numpy as np
from src.io import read_json,write_json,verify,file_hash
from src.data import load
from src.numerics import detection_metrics

def evaluate(data,seed,run):
    data,run=Path(data),Path(run)
    if (run/'unavailable.json').exists():return read_json(run/'unavailable.json')
    verify(read_json(run/'complete.json'))
    view,_,_=load(data,seed);q=view.roles['Q']
    with np.load(run/'scores.npz',allow_pickle=False) as f:
        if f['node_ids'].tolist()!=view.graph.node_ids:raise ValueError('Prediction row order differs')
        scores=f['score'].copy();logits=f['logits'].copy() if 'logits' in f else None
    verify(read_json(data/'sealed/manifest.json'),data/'sealed')
    with np.load(data/'sealed/truth.npz',allow_pickle=False) as f:
        if f['node_ids'].tolist()!=view.graph.node_ids:raise ValueError('Truth row order differs')
        binary=f['is_ood'];labels=f['class_targets']
        metrics=detection_metrics(scores[q],binary[q])
        if logits is not None:
            rows=q[binary[q]==0];metrics['ID_accuracy']=float(np.mean(logits[rows,:len(view.id_definitions)].argmax(1)==labels[rows]))
        if (run/'exposure.json').exists():
            selected=np.asarray(read_json(run/'exposure.json')['rows'],dtype=np.int64);c=np.r_[view.roles['A'],view.roles['B']]
            if not set(selected)<=set(c):raise ValueError('Exposure escaped C')
            tp=int(binary[selected].sum());precision=tp/len(selected) if len(selected) else 0.;recall=tp/int(binary[c].sum()) if binary[c].sum() else 0.
            metrics.update(exposure_precision=precision,exposure_recall=recall,F1_OOD=2*precision*recall/(precision+recall) if precision+recall else 0.)
    report=dict(seed=seed,metrics=metrics,units='fractions, multiply by 100 for percent',AUPR_positive='ID',predictions_sha256=file_hash(run/'scores.npz'))
    write_json(run/'metrics.json',report);return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--run',type=Path,required=True);a=p.parse_args();print(evaluate(a.data,a.seed,a.run))
if __name__=='__main__':main()
