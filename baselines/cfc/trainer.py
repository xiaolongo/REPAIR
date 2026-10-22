import csv,time
import numpy as np
import torch
import torch.nn.functional as F
from src.io import read_json as read, write_json as write, file_hash as sha, _save_checkpoint, verify as verify_legacy
from src.io import public_inputs_only
from src.numerics import seed_everything,IDStop,propagate,parameter_hash,_cpu,_rng_state
from baselines.cfc.loss import denoise,mixup_loss
from baselines.cfc import author
def author_module(unused): return author


def snapshot(net, optimizer, epoch, ce):
    return {"epoch": epoch, "V_CE": ce, "model": _cpu(net.state_dict()),
            "optimizer": _cpu(optimizer.state_dict()), "rng": _rng_state(),
            "parameter_hash": parameter_hash(net)}

def train(slug,seed):
    dest=model_dir(slug,seed)
    if (dest/'complete.json').exists():verify_legacy(read(dest/'complete.json'));return
    with public_inputs_only():
        x,edge,ids,y,roles,op,pod,c=context(slug,seed);real,val,q=(roles[k] for k in ['L','V','Q']);author=author_module(VENDOR)
        kept,lp=denoise(author,y.cuda(),edge.cuda(),real,pod,c,3);dest.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest/'denoise.npz',input_rows=pod,retained_rows=kept,label_scores=lp.cpu().numpy())
        seed_everything(seed,True);gx,ge,gy=x.cuda(),edge.cuda(),y.cuda();net=author.GCN_model2(384,512,128,c+1).cuda()
        opt=torch.optim.SGD(net.parameters(),lr=.01,momentum=.9,weight_decay=.0005);initial=snapshot(net,opt,0,None);_save_checkpoint(dest/'initial.pt',initial)
        write(dest/'protocol.json',dict(seed=seed,variant='A',mode='CFC',config=CFG,classes=c,real_ID_rows=real.tolist(),input_OOD_rows=pod.tolist(),training_OOD_rows=kept.tolist(),plan_sha256=sha(OUT/'plan.json'),oracle_graph_edits=False,C_Q_truth_read=False,LLM_scope='existing GPT binary labels, not native end-to-end prompts'))
        hist=[];boundaries=[];best=stop=None;done=False;started=time.monotonic()
        for epoch in range(1,CFG['max_epoch']+1):
            net.train();opt.zero_grad(set_to_none=True);beta=torch.distributions.Beta(1.,1.).sample([]).item()
            retained,_=denoise(author,gy,ge,real,pod,c,3);assert np.array_equal(retained,kept)
            loss,boundary=mixup_loss(net,gx,ge,gy,real,kept,beta,100,c);assert torch.isfinite(loss);loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters());opt.step();net.eval()
            with torch.no_grad():z=net(gx,ge);ce=float(F.cross_entropy(z[val],gy[val]));acc=float((z[val].argmax(1)==gy[val]).double().mean())
            assert np.isfinite(ce)
            if best is None or ce<best['V_CE']:best=snapshot(net,opt,epoch,ce)
            if epoch==1000:stop=IDStop(ce,ce,epoch,epoch,300,.0001)
            elif epoch>1000:_,done=stop.update(epoch,ce)
            row=dict(epoch=epoch,train_CE=float(loss.detach()),V_CE=ce,V_accuracy=acc,beta=beta,mixup_n=len(boundary),best_epoch=best['epoch'],last_improvement=stop.last_improvement if stop else 0)
            hist.append(row);boundaries.append(boundary)
            if epoch==1 or epoch%100==0 or done:
                write(OUT/slug/f'seed{seed}_progress.json',dict(**row,seconds=time.monotonic()-started));print(dict(dataset=slug,seed=seed,**row),flush=True)
            if done:break
        _save_checkpoint(dest/'selected.pt',best);_save_checkpoint(dest/'last.pt',snapshot(net,opt,epoch,ce))
        with (dest/'history.csv').open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=hist[0]);w.writeheader();w.writerows(hist)
        np.savez_compressed(dest/'boundaries.npz',rows=np.asarray(boundaries));net.load_state_dict(best['model']);net.eval()
        with torch.no_grad():
            z=net(gx,ge).cpu();prob=z.double().softmax(1)[:,c]
            loss,boundary=mixup_loss(net,gx,ge,gy,real,kept,.5,100,c)
        write(dest/'loss_probe.json',dict(beta=.5,loss=float(loss),boundary=boundary.tolist()))
        scores={f'score_{k}':propagate(prob,edge,k,.5)[q].numpy() for k in STEPS}
        np.savez_compressed(dest/'scores_locked.npz',Q_rows=q,Q_ids=ids[q],logits=z.numpy(),**scores)
        names=['initial.pt','selected.pt','last.pt','protocol.json','history.csv','boundaries.npz','denoise.npz','loss_probe.json','scores_locked.npz']
        write(dest/'complete.json',dict(selected_epoch=best['epoch'],stop_epoch=epoch,stop_reason='ID_V_patience' if done else 'epoch_cap',new_API_calls=0,seconds=time.monotonic()-started,files={str(dest/n):sha(dest/n) for n in names}))
