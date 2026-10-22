"""ID-only training is shared with REPAIR's M0; detection propagates Energy."""
import torch
from src.pretrain import train
from src.numerics import propagate

def score(logits,edge):
    return propagate(-torch.logsumexp(logits,dim=1),edge,3,.5)
