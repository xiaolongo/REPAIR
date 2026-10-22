import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from scipy.special import expit
from src.io import read_json, write_json, write_npz, file_hash, _save_checkpoint
from src.numerics import make_classifier, classifier_forward, parameter_hash, freeze_backbone, _cpu


class SemanticHead(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.ones((), dtype=torch.float64))
        self.w = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float64))
        self.b = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))

    def forward(self, x, e):
        return self.alpha * e + x @ self.w + self.b

def objective(theta, features, targets, weights, decay):
    scores = features @ theta
    bce = np.dot(weights, np.logaddexp(0., (1 - 2 * targets) * scores))
    gradient = features.T @ (weights * (expit(scores) - targets)) + decay * theta
    value = bce + .5 * decay * np.dot(theta, theta)
    return float(value), gradient

def diagnostics(theta, features, targets, weights, decay):
    value, gradient = objective(theta, features, targets, weights, decay)
    norm = float(np.linalg.norm(gradient))
    penalty = float(.5 * decay * np.dot(theta, theta))
    return {"objective": value, "BCE": value - penalty, "L2_penalty": penalty,
            "gradient_l2": norm, "gradient_inf": float(np.max(np.abs(gradient))),
            "objective_gap_upper_bound": norm * norm / (2 * decay)}

def solve(features, targets, weights, initial, decay, settings):
    if decay <= 0:
        raise ValueError("Positive L2 on every parameter is required for the certificate")
    result = minimize(objective, initial, args=(features, targets, weights, decay),
                      method="L-BFGS-B", jac=True, options={"gtol": settings["gtol"],
                      "ftol": settings["ftol"], "maxiter": settings["maxiter"], "maxls": 50})
    stats = diagnostics(result.x, features, targets, weights, decay)
    stats.update(iterations=int(result.nit), evaluations=int(result.nfev),
                 solver_success=bool(result.success), stop_message=str(result.message))
    if not np.isfinite(result.x).all() or stats["objective_gap_upper_bound"] > settings["max_objective_gap"]:
        raise RuntimeError(f"Convergence certificate failed: {stats}")
    return result.x, stats

def detection_loss(logits,text_projection,alpha,bias,norm,l,pod):
    energy=-torch.logsumexp(logits,dim=1)
    score=alpha*((energy.double()-norm['mean'])/norm['std'])+text_projection+bias
    id_loss=F.softplus(score[l]).mean();ood_loss=F.softplus(-score[pod]).mean()
    return .5*id_loss+.5*ood_loss,id_loss,ood_loss

def fit_scoring_head(x,energy,l,pod,settings,folder,source_hash):
    raw=torch.from_numpy(energy).double()
    norm={'mean':float(raw[l].mean()),'std':float(raw[l].std(correction=0)),'ddof':0}
    assert np.isfinite(norm['std']) and norm['std']>0
    normalized=(raw-norm['mean'])/norm['std']
    features=np.column_stack([normalized.numpy(),x.numpy(),np.ones(len(x))])
    rows=l+pod;y=np.r_[np.zeros(len(l)),np.ones(len(pod))]
    weights=np.r_[np.full(len(l),.5/len(l)),np.full(len(pod),.5/len(pod))]
    start=np.zeros(features.shape[1]);start[0]=1.
    theta,stats=solve(features[rows],y,weights,start,.0005,settings)
    _,other=solve(features[rows],y,weights,np.zeros_like(start),.0005,settings)
    assert abs(stats['objective']-other['objective'])<=settings['max_objective_gap']
    head=SemanticHead(x.shape[1])
    with torch.no_grad():
        head.alpha.fill_(theta[0]);head.w.copy_(torch.from_numpy(theta[1:-1]));head.b.fill_(theta[-1])
    freeze_backbone(head);score=head(x,normalized).numpy()
    assert np.max(np.abs(score-features@theta))<1e-12
    _save_checkpoint(folder/'head.pt',{'model':head.state_dict(),'norm':norm,'source_GNN_hash':source_hash})
    write_json(folder/'fit.json',{'real_ID_rows':l,'pseudo_OOD_rows':pod,'normalization':norm,
               'normalization_scope':'current GNN eval Energy on real L only','source_GNN_hash':source_hash,
               'converged':stats,'second_start':other,'initialization':'alpha1,w0,b0'})
    return head,norm,score

def snapshot(model,optimizer,epoch,objective):
    return {'model':_cpu(model.state_dict()),'structure':model.structure,'optimizer':_cpu(optimizer.state_dict()),
            'epoch':epoch,'objective':objective,'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all()}

def cpu_scores(model,view,s):
    model.cpu().eval()
    with torch.no_grad():
        logits=classifier_forward(model,view.graph.x,'gcn',s,s,.5).logits
        energy=-torch.logsumexp(logits,dim=1)
    return logits.numpy(),energy.numpy()
