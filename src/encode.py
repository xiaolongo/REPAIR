"""Encode original texts in graph row order, using the existing Nomic-384 recipe."""
import argparse
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel,AutoTokenizer
from src.nomic import encode
from src.io import read_json,write_npz,write_json,file_hash

def run(graph,texts,model_path,out,reject_long=False):
    with np.load(graph,allow_pickle=False) as f:ids=f['node_ids'].copy()
    text=read_json(texts);inputs=['classification: '+text[n] for n in ids]
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False);tokenizer.model_max_length=1000000
    net,info=AutoModel.from_pretrained(model_path,local_files_only=True,trust_remote_code=False,use_safetensors=True,rope_parameters=dict(rope_theta=1000.,rope_type='dynamic',factor=2.),attn_implementation='sdpa',dtype=torch.float32,output_loading_info=True)
    if any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys']):raise ValueError(info)
    net.eval().requires_grad_(False).cuda();lengths=np.array(tokenizer(inputs,truncation=False,return_length=True)['length']);order=np.argsort(np.minimum(lengths,8192),kind='stable');features=np.empty((len(ids),384),np.float32)
    if reject_long and (lengths>8192).any():raise ValueError('Generated text exceeds encoder context')
    pos=0
    while pos<len(order):
        size=1
        if lengths[order[pos]]<=2048:
            while size<16 and pos+size<len(order) and lengths[order[pos+size]]<=2048 and (size+1)*min(lengths[order[pos+size]],8192)<=8192:size+=1
        rows=order[pos:pos+size];features[rows],_=encode([inputs[i] for i in rows],tokenizer,net);pos+=size
        if pos%512<size:print(dict(encoded=pos,total=len(ids)),flush=True)
    write_npz(out,node_ids=ids,x=features);write_json(Path(out).with_suffix('.json'),dict(text_sha256=file_hash(texts),max_length=8192,truncated=int((lengths>8192).sum()),prefix='classification: ',postprocess='mean pooling, LayerNorm768, first384, L2'))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--graph',type=Path,required=True);p.add_argument('--texts',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.graph,a.texts,a.model,a.output)
if __name__=='__main__':main()
