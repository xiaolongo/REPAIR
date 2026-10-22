"""Attach verified fixed annotations, preserving failed responses as excluded nodes."""
import argparse
from pathlib import Path
from src.io import read_json,write_json
from src.data_io import records,manifest
from src.data import load

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--annotations',type=Path,required=True,help='Directory containing seed0.json, seed1.json, seed2.json');a=p.parse_args()
    old=read_json(a.data/'manifest.json')
    for seed in [0,1,2]:
        dest=a.data/f'annotations/seed{seed}.json'
        if dest.exists():raise FileExistsError('Annotations already exist; create a separate data bundle')
        write_json(dest,records(a.annotations/f'seed{seed}.json'))
    manifest(a.data,old['source'])
    for seed in [0,1,2]:load(a.data,seed,True)
if __name__=='__main__':main()
