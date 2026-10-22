"""Run one paper baseline and one seed. No annotation API is called here."""
import argparse,os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from src.data import DATASETS,load,model,identity
from src.io import ROOT,read_json,write_json,write_npz,_save_checkpoint,write_csv,lock_run,seal,verify,public_inputs_only,file_hash
from src.numerics import seed_everything,_cpu,propagate,IDStop
from src.pretrain import train as pretrain
from baselines.goe_identifier.loss import exposure_loss,propagation_matrix,gnnsafe_energy
from baselines.llm_ood_oe.loss import loss_terms
from baselines.lgplug.loss import exposure_loss as lg_loss

METHODS=['text_knn','gnnsafe','llm_ood_oe','goe_identifier','grasp','gold','cfc','lect','lgplug']

def finish(out,view,score,logits=None):
    arrays=dict(node_ids=view.graph.node_ids,score=np.asarray(score))
    if logits is not None:arrays['logits']=np.asarray(logits)
    write_npz(out/'scores.npz',**arrays)
    files=['run.json','scores.npz']+[p.name for p in [out/'selected.pt',out/'last.pt',out/'history.csv',out/'exposure.json'] if p.exists()]
    seal(out,files)

def ordinary(method,data,slug,seed,out,m0,weight,max_epochs=None):
    view,s,labels=load(data,seed,method in ['llm_ood_oe','goe_identifier']);x,y=view.graph.x,view.cls_targets;edge=view.graph.edge_index
    if method=='text_knn':
        from baselines.text_knn.model import knn_distance
        if len(view.roles['L'])<10:raise ValueError('Text kNN needs at least 10 labeled nodes')
        z=x.numpy();reference=z[view.roles['L']]
        score=np.concatenate([knn_distance(z[i:i+1024],reference,10) for i in range(0,len(z),1024)])
        finish(out,view,score);return
    if method in ['gnnsafe','grasp','llm_ood_oe']:
        if m0 is None:m0=pretrain(data,slug,seed,out/'M0',max_epochs=max_epochs)
        state=torch.load(m0,weights_only=True,map_location='cpu')
    if method in ['gnnsafe','grasp']:
        net,forward=model(slug,view,seed);net.load_state_dict(state['model']);net.eval()
        with torch.no_grad():z=forward(net,x,'gcn',s,s,.5).logits
        if method=='gnnsafe':
            from baselines.gnnsafe.model import score as gnnsafe_score
            score=gnnsafe_score(z,edge).numpy()
        else:
            from baselines.grasp.author import GRASP
            args=SimpleNamespace(T=1.,tau1=5.,tau2=50.,delta=1.001,st='top',col=False,adj1=False,test=False,dataset=slug,K=3,alpha=.5)
            known=torch.from_numpy(np.r_[view.roles['L'],view.roles['V']]);unknown=torch.from_numpy(np.sort(np.r_[view.roles['A'],view.roles['B'],view.roles['Q']]))
            score=-GRASP(args).detect(z,SimpleNamespace(edge_index=edge,num_nodes=len(x)),known,unknown,torch.empty(0,dtype=torch.long),args).numpy()
        finish(out,view,score,z.numpy());return
    pod=np.flatnonzero(labels==1)
    if method=='lgplug':
        path=Path(data)/'lgplug'/f'seed{seed}'/'exposure.npz'
        verify(read_json(path.parent/'complete.json'),path.parent)
        with np.load(path,allow_pickle=False) as f:
            pod=f['rows'].astype(np.int64)
            if 'node_ids' in f and f['node_ids'].tolist()!=np.asarray(view.graph.node_ids)[pod].tolist():raise ValueError('LG-Plug node IDs differ')
        if not len(pod):
            write_json(out/'unavailable.json',dict(status='N/A',reason='LG-Plug selected an empty exposure pool',training_started=False,metrics=None));return
    if not len(pod) or len(set(pod))!=len(pod) or not set(pod)<=set(np.r_[view.roles['A'],view.roles['B']]):raise ValueError('Invalid or empty exposure pool')
    write_json(out/'exposure.json',dict(rows=pod))
    seed_everything(seed);net,forward=model(slug,view,seed)
    if method=='llm_ood_oe':net.load_state_dict(state['model'])
    net.cuda();opt=torch.optim.Adam(net.parameters(),lr=.001 if method=='llm_ood_oe' else .01,weight_decay=.0005)
    gx,gy,gs=x.cuda(),y.cuda(),s.cuda();gp=propagation_matrix(edge,len(x)).cuda();l,v=view.roles['L'],view.roles['V']
    if method in ['goe_identifier','lgplug']:l=np.sort(l);pod=np.sort(pod)
    def terms():
        z=forward(net,gx,'gcn',gs,gs,.5).logits
        if method=='llm_ood_oe':parts=loss_terms(z,gy,l,pod,gp,3,weight,-5.,-1.)
        elif method=='goe_identifier':parts=exposure_loss(z,gy,l,pod,weight)
        else:
            p=lg_loss(gnnsafe_energy(z,gp,3,.5),l,pod,-5.,-1.);ce=F.cross_entropy(z[l],gy[l]);parts=dict(classification=ce,exposure=p['total'],total=ce+weight*p['total'])
        return z,parts
    def measure():
        net.eval()
        with torch.no_grad():
            z,parts=terms();ce=float(F.cross_entropy(z[v],gy[v]));obj=ce
            if method=='llm_ood_oe':obj=float(parts['CE'].double()+weight*(parts['ID_reg'].double()+parts['OOD_reg'].double())+.00025*sum(p.double().square().sum() for p in net.parameters()))
        return ce,obj
    best=None;stop=None;stale=0;history=[]
    def snapshot(epoch,ce,obj):return dict(model=_cpu(net.state_dict()),epoch=epoch,V_CE=ce,objective=obj,structure=net.structure)
    if method=='llm_ood_oe':
        ce,obj=measure();best=snapshot(0,ce,obj);history.append(dict(epoch=0,V_CE=ce,objective=obj))
    maximum=max_epochs or (2000 if method=='llm_ood_oe' else 200 if method=='goe_identifier' else 5000)
    for epoch in range(1,maximum+1):
        net.train();opt.zero_grad(set_to_none=True);_,parts=terms()
        if not torch.isfinite(parts['total']):raise ValueError('Nonfinite loss')
        parts['total'].backward();opt.step();ce,obj=measure()
        if best is None or obj<best['objective']:best=snapshot(epoch,ce,obj);stale=0
        else:stale+=1
        history.append(dict(epoch=epoch,V_CE=ce,objective=obj))
        if epoch%100==0:print(dict(method=method,**history[-1]),flush=True)
        if method=='llm_ood_oe' and stale==20:break
        if method=='lgplug':
            if epoch==1000:stop=IDStop(ce,ce,epoch,epoch)
            elif epoch>1000 and stop.update(epoch,ce)[1]:break
    last=snapshot(epoch,ce,obj);chosen=last if method=='goe_identifier' else best
    _save_checkpoint(out/'last.pt',last);_save_checkpoint(out/'selected.pt',chosen);write_csv(out/'history.csv',history)
    net.load_state_dict(chosen['model']);net.eval()
    # Preserve server CUDA scoring for GOE/LG; LLM-OOD OE used CPU scoring.
    if method=='llm_ood_oe':net.cpu();gx,gs=x,s;gp=propagation_matrix(edge,len(x))
    with torch.no_grad():z=forward(net,gx,'gcn',gs,gs,.5).logits;score=gnnsafe_energy(z,gp,3,.5).cpu().numpy()
    finish(out,view,score,z.cpu().numpy())

def special(method,data,slug,seed,out):
    view,s,labels=load(data,seed,method=='cfc');x,edge,ids,y=view.graph.x,view.graph.edge_index,np.asarray(view.graph.node_ids),view.cls_targets
    context=lambda *args:(x,edge,ids,y,view.roles,s,np.flatnonzero(labels==1),len(view.id_definitions))
    if method=='cfc':
        from baselines.cfc import trainer as t
        t.context=context;t.OUT=out;t.VENDOR=None;t.model_dir=lambda *args:out;t.STEPS=(3,)
        t.CFG=dict(max_epoch=5000,min_epoch=1000,patience=300,min_delta=.0001,lr=.01,momentum=.9,weight_decay=.0005)
        write_json(out/'plan.json',dict(method='CFC-A',config=t.CFG));t.train(slug,seed)
        with np.load(out/'scores_locked.npz',allow_pickle=False) as f:
            score=np.full(len(ids),np.nan);score[view.roles['Q']]=f['score_3'];z=f['logits'].copy()
    elif method=='lect':
        from baselines.lect import trainer as t
        from baselines.cfc.trainer import snapshot
        import src.io as io
        from src.numerics import _restore_rng
        cfg=read_json(ROOT/'configs/lect.json');t.CFG=cfg;t.STEPS=(3,);t.ROOT=out;t.OUT=out;t.DATA=Path(data).resolve();t.DEST=out
        r=SimpleNamespace(context=context,public_inputs_only=public_inputs_only,seed_everything=seed_everything,snapshot=snapshot,IDStop=IDStop,_restore_rng=_restore_rng,_save_checkpoint=_save_checkpoint,write=write_json,read=read_json,verify=lambda rec:verify(rec,Path(data)/'lect'/f'seed{seed}'),sha=file_hash,propagate=propagate)
        t.r=r;t.train(slug,seed)
        with np.load(out/'scores_locked.npz',allow_pickle=False) as f:
            score=np.full(len(ids),np.nan);score[view.roles['Q']]=f['centered_3'];z=f['logits'].copy()
    else:
        from baselines.gold import trainer as t
        t.BASE=ROOT/'baselines/gold';t.VENDOR=t.BASE/'vendor/author/gold';t.OUT=out.parent;t.REPORT=out;t.SLUG=slug;t.PRESET='arxiv' if slug=='arxiv' else 'cora';t.ORDER=[slug];t.NAMES=[slug]
        v=SimpleNamespace(graph=view.graph,roles=view.roles,cls_targets=y,num_classes=len(view.id_definitions))
        t.context=lambda seed:(Path(data),v,None)
        write_json(out/'plan.json',dict(method='GOLD',preset=t.PRESET+'/label',selection='author final'))
        t.run(seed)
        with np.load(out/'scores_locked.npz',allow_pickle=False) as f:
            score=np.full(len(ids),np.nan);score[view.roles['Q']]=f['score'];z=f['class_logits'].copy()
    finish(out,view,score,z)

def run(method,data,slug,seed,out,m0=None,weight=None,max_epochs=None):
    out=Path(out).resolve();data=Path(data).resolve();torch.set_num_threads(4)
    # Other datasets keep their existing primary .01 weight.
    if weight is None:weight=.05 if method=='goe_identifier' and slug=='sportsfit' else .01
    auxiliary=data/method/f'seed{seed}'/'complete.json' if method in ['lect','lgplug'] else None
    lock_run(out,identity(data,slug,seed,method,weight=weight,max_epochs=max_epochs,m0_sha256=file_hash(m0) if m0 else None,auxiliary_sha256=file_hash(auxiliary) if auxiliary else None))
    if (out/'complete.json').exists() and (out/'scores.npz').exists():verify(read_json(out/'complete.json'));return
    with public_inputs_only():
        if method in ['cfc','lect','gold']:special(method,data,slug,seed,out)
        else:ordinary(method,data,slug,seed,out,m0,weight,max_epochs)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--dataset',choices=DATASETS,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--seed',type=int,choices=[0,1,2],default=0);p.add_argument('--output',type=Path,required=True);p.add_argument('--m0',type=Path);p.add_argument('--weight',type=float)
    a=p.parse_args()
    if a.method=='gold' and a.output.name!=f'seed{a.seed}':p.error('GOLD output must end in seed0 / seed1 / seed2')
    run(a.method,a.data,a.dataset,a.seed,a.output,a.m0,a.weight)
if __name__=='__main__':main()
