"""Public dataset manifests and fixed annotation records."""

from src.io import read_json,write_json,file_hash

def manifest(out,source):
    files={p.relative_to(out).as_posix():file_hash(p) for p in out.rglob('*') if p.is_file() and not {'sealed','lect','lgplug'}.intersection(p.relative_to(out).parts) and p.name!='manifest.json'}
    write_json(out/'manifest.json',dict(source=source,files=files))

def records(source):
    result=[]
    for r in read_json(source):
        result.append(dict(node_id=r['node_id'],status=r['status'],annotation=r.get('annotation')))
    return result
