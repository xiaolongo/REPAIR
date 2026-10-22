from pathlib import Path
import csv,time
import numpy as np
import torch
import torch.nn.functional as F
from baselines.lect.model import LECT,losses,supervision


def augment(slug,seed,x,edge,roles):
    folder=DATA/'lect'/f'seed{seed}';r.verify(r.read(folder/'complete.json')['files']);nodes=r.read(folder/'nodes.json')
    with np.load(folder/'features.npz',allow_pickle=False) as f:gen=f['x'].copy()
    pairs,triples=supervision(roles['L'],edge.numpy(),nodes,len(x))
    edges=np.unique(np.concatenate([edge.numpy(),pairs.T,pairs.T[::-1]],axis=1),axis=1)
    n=len(x)+len(gen);src=np.r_[edges[0],np.arange(n)];dst=np.r_[edges[1],np.arange(n)];degree=np.bincount(dst,minlength=n).astype(np.float32)
    values=torch.from_numpy(degree[dst]*degree[src]).rsqrt()
    op=torch.sparse_coo_tensor(torch.from_numpy(np.array([dst,src])),values,(n,n)).coalesce()
    return torch.cat([x,torch.from_numpy(gen)]),op,pairs,triples,edges

def train(slug,seed,probe=False):
    dest=DEST
    if not probe and (dest/'complete.json').exists():r.verify(r.read(dest/'complete.json')['files']);return
    with r.public_inputs_only():
        x,e,ids,y,roles,original,_,classes=r.context(slug,seed);aug,s,pairs,triples,edges=augment(slug,seed,x,e,roles)
        gx,gs,x0,s0,gy=aug.cuda(),s.cuda(),x.cuda(),original.cuda(),y.cuda();l,v,q=(roles[k] for k in ['L','V','Q'])
        r.seed_everything(seed,True);net=LECT(384,classes,CFG).cuda();opt=torch.optim.Adam(net.parameters(),lr=CFG['lr'],weight_decay=CFG['weight_decay']);rng=np.random.default_rng(seed)
        initial=r.snapshot(net,opt,0,None);best=stop=None;hist=[];done=False;start=1;started=time.monotonic()
        if not probe:
            dest.mkdir(parents=True,exist_ok=True)
            journal=dest/'journal.pt'
            if journal.exists():
                saved=torch.load(journal,map_location='cpu',weights_only=True);net.load_state_dict(saved['current']['model']);opt.load_state_dict(saved['current']['optimizer']);r._restore_rng(saved['current']['rng']);rng.bit_generator.state=saved['sampling_rng'];hist=saved['history'];best=saved['best'];start=saved['current']['epoch']+1
                if saved['stop']:stop=r.IDStop(**saved['stop'])
                done=bool(stop and hist[-1]['epoch']-stop.last_improvement>=stop.patience)
                epoch=hist[-1]['epoch'];ce=hist[-1]['V_CE']
            else:r._save_checkpoint(dest/'initial.pt',initial)
            r.write(dest/'protocol.json',dict(config=CFG,classes=classes,seed=seed,centering_only_at_scoring=True,primary_K=3,alpha=.5,source_model_sha256=r.sha(Path(__file__).with_name('model.py')),input_sha256=r.sha(DATA/'manifest.json'),features_sha256=r.sha(DATA/'lect'/f'seed{seed}'/'complete.json'),C_Q_truth_read=False))
            np.savez_compressed(dest/'supervision.npz',pairs=pairs,triples=triples,edges=edges)
        for epoch in range(start,start if done else 2 if probe else CFG['max_epoch']+1):
            net.train();opt.zero_grad(set_to_none=True);ps=torch.as_tensor(pairs[rng.integers(len(pairs),size=CFG['pair_samples'])],device='cuda');ts=torch.as_tensor(triples[rng.integers(len(triples),size=CFG['triplet_samples'])],device='cuda')
            parts=losses(net(gx,gs),gy,l,ps,ts,CFG);assert all(torch.isfinite(z) for z in parts.values());parts['total'].backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters());opt.step();net.eval()
            with torch.no_grad():z=net(x0,s0);ce=float(F.cross_entropy(z[v],gy[v]));acc=float((z[v].argmax(1)==gy[v]).double().mean())
            assert np.isfinite(ce)
            if best is None or ce<best['V_CE']:best=r.snapshot(net,opt,epoch,ce)
            if epoch==CFG['min_epoch']:stop=r.IDStop(ce,ce,epoch,epoch,CFG['patience'],CFG['min_delta'])
            elif epoch>CFG['min_epoch']:_,done=stop.update(epoch,ce)
            row=dict(epoch=epoch,**{k:float(z.detach()) for k,z in parts.items()},V_CE=ce,V_accuracy=acc,best_epoch=best['epoch'],last_improvement=stop.last_improvement if stop else 0);hist.append(row)
            if probe:return dict(passed=True,dataset=slug,peak_bytes=torch.cuda.max_memory_allocated(),probe_not_used_as_model=True)
            if epoch%100==0 or done:
                r._save_checkpoint(journal,dict(current=r.snapshot(net,opt,epoch,ce),best=best,sampling_rng=rng.bit_generator.state,history=hist,stop=vars(stop) if stop else None))
                r.write(OUT/slug/f'seed{seed}_progress.json',dict(**row,seconds=time.monotonic()-started));print(dict(dataset=slug,seed=seed,**row),flush=True)
            if done:break
        r._save_checkpoint(dest/'selected.pt',best);r._save_checkpoint(dest/'last.pt',r.snapshot(net,opt,epoch,ce))
        with (dest/'history.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=hist[0]);w.writeheader();w.writerows(hist)
        net.load_state_dict(best['model']);net.eval()
        with torch.no_grad():
            z=net(x0,s0).cpu().double();za=net(gx,gs).cpu().double();probe_parts=losses(za,y,l,torch.from_numpy(pairs[:1000]),torch.from_numpy(triples[:1000]),CFG)
        arrays=dict(Q_ids=ids[q],Q_rows=q,logits=z.numpy())
        for name,zz in [('raw',z),('centered',z-z.mean(1,keepdim=True))]:
            energy=-torch.logsumexp(zz,1)
            for k in STEPS:arrays[f'{name}_{k}']=r.propagate(energy,e,k,.5)[q].numpy()
        np.savez_compressed(dest/'scores_locked.npz',**arrays);r.write(dest/'loss_probe.json',{k:float(z) for k,z in probe_parts.items()})
        names=['initial.pt','selected.pt','last.pt','history.csv','supervision.npz','protocol.json','scores_locked.npz','loss_probe.json']
        r.write(dest/'complete.json',dict(selected_epoch=best['epoch'],stop_epoch=epoch,stop_reason='ID_V_patience' if done else 'epoch_cap',files={str(dest/n):r.sha(dest/n) for n in names}))
