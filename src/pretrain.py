"""ID-only initialization, selected by minimum ID-validation cross entropy."""
from pathlib import Path
import torch
import torch.nn.functional as F
from src.data import load,model,identity
from src.io import ROOT,read_json,write_npz,write_csv,lock_run,seal,verify,public_inputs_only,_save_checkpoint
from src.numerics import seed_everything,parameter_hash,_cpu,propagate

def train(data,slug,seed,out,device='cuda',max_epochs=None):
    out=Path(out).resolve();cfg=read_json(ROOT/'configs/datasets.json')[slug]
    if max_epochs is not None:cfg['max_epochs']=max_epochs
    lock_run(out,identity(data,slug,seed,'GNNSafe',config=cfg))
    if (out/'complete.json').exists():verify(read_json(out/'complete.json'));return out/'M0.pt'
    with public_inputs_only():
        view,s,_=load(data,seed);seed_everything(seed);net,forward=model(slug,view,seed);net.to(device)
        opt=torch.optim.Adam(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        x,y,op=view.graph.x.to(device),view.cls_targets.to(device),s.to(device);l,v=view.roles['L'],view.roles['V']
        def state(epoch):return dict(epoch=epoch,model=_cpu(net.state_dict()),parameter_hash=parameter_hash(net),structure=net.structure)
        _save_checkpoint(out/'initial.pt',state(0));best=None;stale=0;history=[]
        for epoch in range(1,cfg['max_epochs']+1):
            net.train();opt.zero_grad(set_to_none=True);z=forward(net,x,'gcn',op,op,.5).logits;loss=F.cross_entropy(z[l],y[l])
            if not torch.isfinite(loss):raise ValueError('Nonfinite pretraining loss')
            loss.backward();opt.step();net.eval()
            with torch.no_grad():z=forward(net,x,'gcn',op,op,.5).logits;ce=float(F.cross_entropy(z[v],y[v]))
            if best is None or ce<best['V_CE']:best=dict(**state(epoch),V_CE=ce);stale=0
            else:stale+=1
            history.append(dict(epoch=epoch,L_CE=float(loss.detach()),V_CE=ce,best_epoch=best['epoch'],stale=stale))
            if epoch%100==0:print(history[-1],flush=True)
            if stale==cfg['patience']:break
        _save_checkpoint(out/'last.pt',state(epoch));_save_checkpoint(out/'M0.pt',best);write_csv(out/'history.csv',history)
        net.cpu().load_state_dict(best['model']);net.eval()
        with torch.no_grad():z=forward(net,view.graph.x,'gcn',s,s,.5).logits;e=-torch.logsumexp(z,1)
        write_npz(out/'scores.npz',node_ids=view.graph.node_ids,logits=z.numpy(),energy=e.numpy(),score=propagate(e,view.graph.edge_index,3,.5).numpy())
        seal(out,['run.json','M0.pt','scores.npz','history.csv'])
    return out/'M0.pt'
