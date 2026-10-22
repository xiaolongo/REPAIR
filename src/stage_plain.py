import numpy as np
import torch
import torch.nn.functional as F
from src.io import read_json,write_json,write_npz,file_hash,_save_checkpoint,write_csv
from src.numerics import parameter_hash,seed_everything
from src.head import fit_scoring_head,detection_loss,snapshot
from src.numerics import make_classifier,classifier_forward
from src.head import cpu_scores

def train_stage(view,s,start,pod,settings,cfg,folder,seed,fold=None):
    if (folder/'complete.json').exists():
        done=read_json(folder/'complete.json')
        for p,h in done['files'].items():assert file_hash(p)==h
        return
    l,v=view.roles['L'].tolist(),view.roles['V'].tolist()
    allowed=view.roles[fold] if fold else np.r_[view.roles['A'],view.roles['B']]
    assert pod and len(set(pod))==len(pod) and set(pod)<=set(allowed) and set(pod).isdisjoint(l)
    seed_everything(seed,deterministic=True)
    source=torch.load(start,weights_only=True,map_location='cpu')
    model=make_classifier(view.graph.x.shape[1],128,len(view.id_definitions),seed);model.load_state_dict(source['model'])
    initial_hash=parameter_hash(model)
    before_z,before_e=cpu_scores(model,view,s)
    write_npz(folder/'before_scores.npz',logits=before_z,energy=before_e)
    head,norm,_=fit_scoring_head(view.graph.x.double(),before_e,l,pod,settings,folder/'pre_head',initial_hash)
    head_hash=parameter_hash(head);model.cuda();head.cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    continued='optimizer' in source
    if continued:
        optimizer.load_state_dict(source['optimizer'])
        torch.set_rng_state(source['rng_cpu']);torch.cuda.set_rng_state_all(source['rng_cuda'])
    else:seed_everything(seed,deterministic=True)
    assert all(g['lr']==cfg['lr'] and g['weight_decay']==cfg['weight_decay'] for g in optimizer.param_groups)
    x,y,op=view.graph.x.cuda(),view.cls_targets.cuda(),s.cuda()
    projection=(view.graph.x.double().cuda()@head.w).detach()

    def forward_terms():
        z=classifier_forward(model,x,'gcn',op,op,cfg['dropout']).logits
        ce=F.cross_entropy(z[l],y[l])
        binary,inside,outside=detection_loss(z,projection,head.alpha,head.b,norm,l,pod)
        penalty=.5*cfg['weight_decay']*sum(p.double().square().sum() for p in model.parameters())
        return z,ce,binary,inside,outside,penalty

    def measure():
        model.eval()
        with torch.no_grad():
            z,ce,binary,inside,outside,penalty=forward_terms()
            total=ce+binary+penalty
            assert torch.isfinite(total)
            return {'objective':float(total),'CE':float(ce),'ID_BCE':float(inside),'OOD_BCE':float(outside),
                    'L2':float(penalty),'V_CE':float(F.cross_entropy(z[v],y[v])),
                    'V_accuracy':float((z[v].argmax(1)==y[v]).double().mean())}

    stats=measure();best=snapshot(model,optimizer,0,stats['objective']);initial=snapshot(model,optimizer,0,stats['objective'])
    history=[{'epoch':0,**stats,'best_epoch':0,'stale':0}];stale=0
    for epoch in range(1,cfg['max_epochs']+1):
        model.train();optimizer.zero_grad(set_to_none=True)
        _,ce,binary,_,_,_=forward_terms()
        loss=ce+binary;assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert all(p.grad is None and not p.requires_grad for p in head.parameters())
        optimizer.step()
        stats=measure()
        if stats['objective']<best['objective']:
            best=snapshot(model,optimizer,epoch,stats['objective']);stale=0
        else:stale+=1
        history.append({'epoch':epoch,**stats,'best_epoch':best['epoch'],'stale':stale})
        if epoch%200==0:
            print({'stage':str(folder),'epoch':epoch,'best':best['epoch'],'joint_objective':stats['objective'],
                   'V_accuracy':stats['V_accuracy']},flush=True)
        if stale==cfg['patience']:break
    last=snapshot(model,optimizer,epoch,stats['objective'])
    assert parameter_hash(head)==head_hash
    _save_checkpoint(folder/'initial.pt',initial);_save_checkpoint(folder/'best.pt',best);_save_checkpoint(folder/'last.pt',last)
    write_csv(folder/'history.csv',history)
    model.load_state_dict(best['model']);after_z,after_e=cpu_scores(model,view,s)
    current_hash=parameter_hash(model)
    _,after_norm,scores=fit_scoring_head(view.graph.x.double(),after_e,l,pod,settings,folder/'post_head',current_hash)
    write_npz(folder/'scores.npz',logits=after_z,energy=after_e,score=scores)
    delta=max(float((best['model'][k]-initial['model'][k]).abs().max()) for k in initial['model'])
    protocol={'seed':seed,'fold':fold,'start':str(start),'start_hash':file_hash(start),'optimizer_continued':continued,
              'real_ID_rows':l,'pseudo_OOD_rows':pod,'config':cfg,'head_before_hash':head_hash,'head_frozen_verified':True,
              'before_GNN_hash':initial_hash,'after_GNN_hash':current_hash,'before_normalization':norm,'after_normalization':after_norm,
              'selected_epoch':best['epoch'],'stop_epoch':epoch,'stop_reason':'joint_objective_patience' if stale==cfg['patience'] else 'epoch_cap',
              'GNN_max_parameter_change':delta,'Energy_max_change':float(np.max(np.abs(after_e-before_e))),
              'C_Q_truth_read':False,'V_used_for_stopping':False}
    write_json(folder/'protocol.json',protocol)
    files=[folder/n for n in ('initial.pt','best.pt','last.pt','history.csv','before_scores.npz','scores.npz','protocol.json',
                             'pre_head/head.pt','pre_head/fit.json','post_head/head.pt','post_head/fit.json')]
    write_json(folder/'complete.json',{'selected_epoch':best['epoch'],'stop_epoch':epoch,'files':{str(p):file_hash(p) for p in files}})
    print({'stage':str(folder),'selected_epoch':best['epoch'],'stop_epoch':epoch,'GNN_change':delta,
           'Energy_change':protocol['Energy_max_change']},flush=True)
