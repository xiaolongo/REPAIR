import argparse,os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
from src.io import ROOT,read_json
from src.data import load

def main():
    p=argparse.ArgumentParser(description='LG-Plug graph-text alignment, no LLM queries');p.add_argument('--data',type=Path,required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from baselines.lgplug import alignment
    cfg=read_json(ROOT/'configs/lgplug_alignment.json');cfg.update(tokenizer_path=str(a.tokenizer.resolve()),data_source=str(a.data.resolve()))
    alignment.ROOT=ROOT;alignment.load_training_view=lambda _:load(a.data,a.seed)[0]
    alignment.run_seed(a.seed,cfg,a.output.resolve()/f'seed{a.seed}',a.output.resolve())
if __name__=='__main__':main()
