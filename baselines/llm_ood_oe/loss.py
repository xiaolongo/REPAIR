"""GNNSafe residual propagation within separately averaged energy-margin OE."""
import torch
import torch.nn.functional as F


def propagated_energy(logits, matrix, steps):
    energy = -torch.logsumexp(logits, dim=1)
    for _ in range(steps):
        energy = .5*energy+.5*torch.sparse.mm(matrix, energy[:, None])[:, 0]
    return energy


def loss_terms(logits, labels, real_id, pseudo_ood, matrix, steps, lam, m_in=-5., m_out=-1.):
    ce = F.cross_entropy(logits[real_id], labels[real_id])
    energy = propagated_energy(logits, matrix, steps)
    if lam:
        if not len(pseudo_ood): raise ValueError('OE requires pseudo-OOD nodes')
        inside = F.relu(energy[real_id]-m_in).square().mean()
        outside = F.relu(m_out-energy[pseudo_ood]).square().mean()
    else:
        inside = outside = energy.new_zeros(())
    return {'CE': ce, 'ID_reg': inside, 'OOD_reg': outside, 'total': ce+lam*(inside+outside)}
