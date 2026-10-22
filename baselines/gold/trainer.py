import argparse,ast,hashlib,json,math,os,random,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_sparse import SparseTensor,matmul
from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree
from src.io import read_json,write_json,write_npz,file_hash,verify,public_inputs_only


def cpu(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu(v) for v in value]
    return value

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(cpu(value),tmp);tmp.replace(path)

def progress(folder,**info):
    folder.mkdir(parents=True,exist_ok=True);value={**info,'time':time.time()};tmp=folder/'progress.tmp'
    tmp.write_text(json.dumps(value),encoding='utf-8');tmp.replace(folder/'progress.json');print(value,flush=True)

def definitions():
    manifest=read_json(BASE/'vendor/source_manifest.json')
    for name,value in manifest['files'].items():assert file_hash(BASE/'vendor/author'/name)==value['sha256']
    env={'torch':torch,'nn':nn,'F':F,'np':np,'math':math,'random':random,'argparse':argparse,
        'GCNConv':GCNConv,'SparseTensor':SparseTensor,'matmul':matmul,'degree':degree}
    specs={'backbone.py':{'GCN','MLP'},'gnnsafe.py':{'GNNSafe'},
        'diffusion_util.py':{'SinusoidalPositionEmbeddings','DenoiseNN','extract','q_sample','p_losses','linear_beta_schedule','p_sample','draw_samples'},
        'gold_utils.py':{'fix_seed','dataset_params','initialise_setup','get_energy'},'parse.py':{'parser_add_main_args'}}
    for filename,names in specs.items():
        tree=ast.parse((VENDOR/filename).read_text(encoding='utf-8'));nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
        assert {n.name for n in nodes}==names
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(VENDOR/filename),'exec'),env)
    return env

def run(seed):
    folder=OUT/f'seed{seed}'
    if (folder/'complete.json').exists():verify(read_json(folder/'complete.json'));return
    began=time.perf_counter();env=definitions();history=[];warm_history=[];gen_history=[];best=float('inf')
    source,view,_=context(seed)
    data=SimpleNamespace(x=view.graph.x,edge_index=view.graph.edge_index,y=view.cls_targets[:,None],num_nodes=len(view.graph.node_ids),
        node_idx=torch.as_tensor(np.r_[view.roles['L'],view.roles['V']]),splits={key:torch.as_tensor(view.roles[role]) for key,role in [('train','L'),('valid','V'),('test','Q')]})
    unknown=SimpleNamespace(x=data.x,edge_index=data.edge_index,y=torch.full_like(data.y,-100),
        node_idx=torch.as_tensor(np.sort(np.r_[view.roles['A'],view.roles['B'],view.roles['Q']])))

    def initialise(args):
        # Same model construction, seed call, criterion and train state as the
        # author's initialise; only dataset creation/splitting are supplied here.
        env['fix_seed'](args.seed);device=torch.device('cuda:0')
        model=env['GNNSafe'](data.x.shape[1],view.num_classes,args).to(device);model.train()
        return model,device,nn.NLLLoss(),None,None,data,unknown,unknown
    env['initialise']=initialise
    original_setup=env['initialise_setup']

    def setup(lr,args,use_detector=True):
        returned=original_setup(lr,args,use_detector)
        model,device,criterion,_,_,_,_,_,den,den_opt,warm_opt,det,joint_opt,_,_,_,_,params=returned
        # Capture source-native initialization, including its second reset on GPU.
        save(folder/'initial.pt',{'classifier':model.state_dict(),'detector':det.state_dict(),'denoiser':den.state_dict()})
        write_json(folder/'protocol.json',{'actual_dataset':NAMES[ORDER.index(SLUG)],'author_parameter_preset':PRESET+'/label','seed':seed,
            'author_args':vars(args),'author_params':params,'public_manifest_sha256':file_hash(source/'manifest.json'),
            'input_dim':384,'ID_classes':view.num_classes,'L':len(view.roles['L']),'V':len(view.roles['V']),'Q':len(view.roles['Q']),'generator_fit_nodes':len(view.graph.node_ids),
            'main_result':'last classifier and detector, exactly as original entrypoint','auxiliary':'ID-V best recorded but not used as main',
            'data_adapter_only':True,'training_source_AST_changed':False,'per_round_Q_evaluation_deferred':True,'C_Q_truth_read':False,
            'torch':torch.__version__,'torch_deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'sources':{str(p):file_hash(p) for p in [Path(__file__),BASE/'vendor/source_manifest.json',source/'manifest.json',REPORT/'plan.json']}})
        def warm_hook(opt,args_,kwargs):
            warm_history.append({'step':len(warm_history)+1,'CE':float(env['id_sup_loss'].detach()),'encoder_training':model.training})
            if len(warm_history)==params['disr_iter']:
                save(folder/'warmup.pt',{'classifier':model.state_dict(),'detector':det.state_dict()})
                progress(folder,stage='warmup_completed',updates=len(warm_history))
        def gen_hook(opt,args_,kwargs):
            step=len(gen_history)+1
            gen_history.append({'step':step,'after_round':int(env.get('epoch',0)),'loss':float(env['diffusion_loss'].detach()),
                'fit_nodes':int(env['gen_data'].shape[0]),'encoder_training':model.training,'denoiser_training':den.training})
            assert np.isfinite(gen_history[-1]['loss'])
            if step%(params['denoise_iter']-1)==0:progress(folder,stage='generator_fit_completed',updates=step,after_round=int(env.get('epoch',0)))
        def joint_hook(opt,args_,kwargs):
            item={'step':len(history)+1,'round':int(env['epoch']),'local_step':int(env['main_iter'])+1,
                'encoder_training':model.training,'detector_training':det.training,
                **{key:float(env[key].detach()) for key in ['id_sup_loss','reg_loss','mlp_sup_loss','mlp_reg_loss','loss']}}
            assert all(np.isfinite(item[key]) for key in ['id_sup_loss','reg_loss','mlp_sup_loss','mlp_reg_loss','loss'])
            history.append(item)
        warm_opt.register_step_post_hook(warm_hook);den_opt.register_step_post_hook(gen_hook);joint_opt.register_step_post_hook(joint_hook)
        # Observe the existing V forward at the end of an inner update. The
        # criterion input identifies that call without adding another forward.
        def criterion_hook(module,inputs,output):
            nonlocal best
            if len(inputs[1])==len(view.roles['V']) and history:
                row=history[-1]
                if 'V_CE' in row:return
                row['V_CE']=float(output.detach());row['V_accuracy']=float((inputs[0].argmax(1)==inputs[1]).float().mean())
                if row['V_CE']<best:
                    best=row['V_CE'];save(folder/'aux_IDV.pt',{'classifier':model.state_dict(),'detector':det.state_dict(),'step':row['step'],'V_CE':best})
        criterion.register_forward_hook(criterion_hook)
        return returned
    env['initialise_setup']=setup

    def evaluate(model,detector,ind_data,test_data,syn_energy,eval_func,criterion,args,device,weights,epoch,best_val_loss,**kwargs):
        # These mode changes are the original evaluation callback's only
        # training-relevant side effects. No test labels enter this callback.
        model.eval();detector.eval()
        if not kwargs.get('test',False):
            progress(folder,stage='round_completed',round=epoch,V_CE=history[-1]['V_CE'])
            return None,None,history[-1]['V_CE'],None,None,best_val_loss
        save(folder/'last.pt',{'classifier':model.state_dict(),'detector':detector.state_dict(),'step':len(history)})
        save(folder/'denoiser_last.pt',env['denoise_model'].state_dict())
        q=view.roles['Q']
        with torch.no_grad():
            z,_=model(ind_data,device)
            e,_=env['get_energy'](model,ind_data,ind_data,z,z,args,pos=True,prop=args.prop_gcn,device=device)
            full,_=detector(torch.tensor(np.r_[e,e],device=device).view(-1,1))
            first,second=full[:len(e)],full[len(e):]
            id_score,_=env['get_energy'](model,ind_data,ind_data,first,second,args,pos=False,prop=args.prop_mlp,device=device)
        write_npz(folder/'scores_locked.npz',Q_rows=q,Q_ids=np.asarray(view.graph.node_ids)[q],
            class_logits=z.cpu().numpy(),gnn_energy=e,detector_logits=first.cpu().numpy(),score=-id_score[q],pred=z[q].argmax(1).cpu().numpy())
        return None,None,None,None,None,None
    env['evaluate_performance']=evaluate
    args=['gold.py','--backbone','gcn','--dataset',PRESET,'--ood_type','label','--mode','detect','--use_bn','--use_prop','--use_reg','--m_in','-4','--m_out','-2','--device','0','--seed',str(seed)]
    tree=ast.parse((VENDOR/'gold.py').read_text(encoding='utf-8'))
    body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
    argv=sys.argv;sys.argv=args
    progress(folder,stage='author_training_started')
    try:exec(compile(ast.Module(body=body,type_ignores=[]),str(VENDOR/'gold.py'),'exec'),env)
    finally:sys.argv=argv
    params=env['params']
    assert len(warm_history)==params['disr_iter'] and len(history)==10*params['iter_num'] and len(gen_history)==10*(params['denoise_iter']-1)
    assert all(r['encoder_training']==(r['local_step']==1) for r in history)
    assert all(r['fit_nodes']==len(view.graph.node_ids) and not r['encoder_training'] and r['denoiser_training'] for r in gen_history)
    assert sorted(set(r['after_round'] for r in gen_history))==[0]+[i for i in range(1,11) if i!=params['epoch_num']]
    write_json(folder/'warmup_history.json',warm_history);write_json(folder/'joint_history.json',history);write_json(folder/'generator_history.json',gen_history)
    files=['protocol.json','initial.pt','warmup.pt','aux_IDV.pt','last.pt','denoiser_last.pt','scores_locked.npz','warmup_history.json','joint_history.json','generator_history.json']
    write_json(folder/'complete.json',{'passed':True,'seed':seed,'C_Q_truth_read':False,'seconds':time.perf_counter()-began,
        'author_loop_executed_verbatim_AST':True,'files':{str(folder/name):file_hash(folder/name) for name in files}})
    progress(folder,stage='completed',seconds=time.perf_counter()-began)
