"""Explicit loopback llama.cpp requests. Never starts/stops a model service."""
import json,threading,urllib.request,urllib.error
from urllib.parse import urlparse
from pathlib import Path
from src.io import read_json,write_json,digest

class LocalJSON:
    def __init__(self,endpoint,model,cache,context):
        if urlparse(endpoint).hostname not in ['127.0.0.1','localhost','::1']:raise ValueError('Only an explicitly supplied loopback model endpoint is supported')
        self.endpoint=endpoint.rstrip('/');self.model=model;self.cache=Path(cache);self.context=context;self.stop=threading.Event()
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def api(self,route,body):
        req=urllib.request.Request(self.endpoint+route,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with self.opener.open(req,timeout=600) as f:return json.load(f)

    def ask_json(self,body,validate,repair,cache_key=None):
        key=cache_key or digest(body);folder=self.cache/key;result=folder/'result.json'
        if result.exists():return read_json(result)
        if self.stop.is_set():raise RuntimeError('Query dispatch stopped')
        for p in folder.glob('*.start.json'):
            if not p.with_name(p.name.replace('.start.','.end.')).exists():raise RuntimeError(f'Unresolved in-flight request. Inspect {p}; no automatic resend.')
        repairs=transports=0;attempts=[];current=dict(body);value=None
        old=sorted(folder.glob('*.end.json'))
        # Replay completed attempts so interruption cannot silently enlarge the retry budget.
        for path in old:
            record=read_json(path);attempts.append(path.stem.split('.')[0])
            if record['status']=='http_error':transports+=1;continue
            try:value=self.validate_response(record['raw'],validate);break
            except (ValueError,KeyError,TypeError,AssertionError):repairs+=1
        if transports>2:raise RuntimeError('HTTP retry budget exhausted')
        if repairs:current={**body,'messages':body['messages']+[dict(role='user',content=repair)]}
        while value is None and repairs<=1:
            if self.stop.is_set():raise RuntimeError('Query dispatch stopped')
            rendered=self.api('/apply-template',{k:current[k] for k in ['messages','chat_template_kwargs']})['prompt']
            count=len(self.api('/tokenize',dict(content=rendered,add_special=True))['tokens'])
            if count+current['max_tokens']+8>self.context:
                self.stop.set();raise ValueError('Context limit exceeded; text will not be truncated')
            idx=len(attempts);attempts.append(str(idx));start=folder/f'{idx:02d}.start.json';end=folder/f'{idx:02d}.end.json'
            write_json(start,dict(body=current,prompt_tokens=count))
            try:raw=self.api('/v1/chat/completions',current)
            except urllib.error.HTTPError as exc:
                write_json(end,dict(status='http_error',code=exc.code));transports+=1
                if exc.code in [429,500,502,503,504] and transports<=2:continue
                self.stop.set();raise
            except Exception:
                self.stop.set();raise  # started ledger remains unresolved
            write_json(end,dict(status='response',raw=raw))
            try:value=self.validate_response(raw,validate)
            except (ValueError,KeyError,TypeError,AssertionError):
                repairs+=1;current={**body,'messages':body['messages']+[dict(role='user',content=repair)]}
        record=dict(status='valid' if value is not None else 'format_error',value=value,attempt_ids=attempts,schema_repairs=min(repairs,1),transport_errors=transports)
        write_json(result,record);return record

    def validate_response(self,raw,validate):
        c=raw['choices'][0]
        if raw['model']!=self.model or c['finish_reason']!='stop' or c['message'].get('reasoning_content'):raise ValueError('Model/finish/thinking mismatch')
        return validate(json.loads(c['message']['content']))
