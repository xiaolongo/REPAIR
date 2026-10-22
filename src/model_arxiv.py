import torch
import torch.nn.functional as F
from src.numerics import make_classifier as plain_classifier, LogitsBundle

def make_classifier(input_dim, hidden_dim, classes, seed):
    assert input_dim == 384 and hidden_dim == 256 and classes == 19
    model = plain_classifier(input_dim, hidden_dim, classes, seed)
    model['bn'] = torch.nn.BatchNorm1d(hidden_dim, eps=1e-5, momentum=.1)
    model.structure.update(BN=True, BN_position='first GCN then BN then ReLU then Dropout',
                           BN_eps=1e-5, BN_momentum=.1)
    return model

def classifier_forward(model, x, kind, s_first, s_second, dropout=.5):
    assert kind == 'gcn'
    h = torch.sparse.mm(s_first, F.linear(x, model['layer0'].weight)) + model['layer0'].bias
    h = F.relu(model['bn'](h))
    z = torch.sparse.mm(s_second, F.linear(F.dropout(h, dropout, model.training), model['layer1'].weight)) + model['layer1'].bias
    return LogitsBundle(z, h)

def cpu_scores(model, view, s):
    model.cpu().eval()
    state = {k:v.clone() for k,v in model['bn'].state_dict().items()}
    with torch.no_grad():
        z = classifier_forward(model, view.graph.x, 'gcn', s, s).logits
        e = -torch.logsumexp(z, dim=1)
    assert all(torch.equal(v, model['bn'].state_dict()[k]) for k,v in state.items())
    return z.numpy(), e.numpy()

def param_items(state):
    return {k:v for k,v in state.items() if not k.startswith('bn.running_') and k != 'bn.num_batches_tracked'}
