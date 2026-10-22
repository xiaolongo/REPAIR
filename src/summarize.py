"""Aggregate three evaluated seeds with sample SD; missing runs remain N/A."""
import argparse
from pathlib import Path
import numpy as np
from src.io import read_json,write_json

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',type=Path,required=True,help='Parent of seed0, seed1, seed2');a=p.parse_args();runs=[]
    for seed in [0,1,2]:
        path=a.runs/f'seed{seed}'/'metrics.json';runs.append(read_json(path)['metrics'] if path.exists() else None)
    summary={}
    for key in ['AUROC','AUPR','FPR95','ID_accuracy','F1_OOD']:
        values=[r[key] for r in runs if r and key in r]
        summary[key]=dict(mean_percent=100*float(np.mean(values)) if len(values)==3 else None,sample_sd_percent=100*float(np.std(values,ddof=1)) if len(values)==3 else None,n_valid=len(values),n_expected=3)
    write_json(a.runs/'summary.json',summary);print(summary)
if __name__=='__main__':main()
