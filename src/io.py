"""Portable local IO. Training never opens the sealed evaluation labels."""
import builtins
import csv
import hashlib
import io
import json
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import patch
import numpy as np
import torch
from src.numerics import digest, array_hash, _cpu

ROOT = Path(__file__).resolve().parents[1]

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4194304),b''): h.update(chunk)
    return h.hexdigest()

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))+'\n',encoding='utf-8')
    temp.replace(path)

def write_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,**arrays)
    temp.replace(path)

def _save_checkpoint(path,value,**unused):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');torch.save(_cpu(value),temp);temp.replace(path)

def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

def verify(record,base=None):
    for name,expected in record.get('files',record).items():
        path=Path(name)
        if not path.is_absolute() and base is not None:path=Path(base)/path
        if file_hash(path)!=expected:raise ValueError(f'Hash mismatch: {path}')

@contextmanager
def public_inputs_only():
    def guard(fn):
        def checked(path,mode='r',*args,**kwargs):
            if isinstance(path,(str,Path)) and ('r' in mode or '+' in mode) and 'sealed' in Path(path).parts:
                raise PermissionError('Training cannot read sealed labels')
            return fn(path,mode,*args,**kwargs)
        return checked
    with patch.object(builtins,'open',guard(builtins.open)),patch.object(io,'open',guard(io.open)):
        yield

def lock_run(out,identity):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    p=out/'run.json'
    if p.exists() and read_json(p)!=identity:raise ValueError('Output belongs to another configuration. Choose a new --output.')
    write_json(p,identity)

def seal(out,names):
    out=Path(out)
    write_json(out/'complete.json',dict(files={str((out/n).resolve()):file_hash(out/n) for n in names},truth_read=False))
