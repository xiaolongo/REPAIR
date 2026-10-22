"""GOE-Identifier loss and GNNSafe scoring, ported from its linked source."""
import torch
import torch.nn.functional as F


def exposure_loss(logits, targets, real_id, pseudo_ood, lam, m_in=-5., m_out=-1.):
    if not len(real_id) or not len(pseudo_ood):
        raise ValueError("GOE needs real ID and pseudo OOD samples")
    # The original boolean masks enumerate nodes in graph-row order.
    real_id = torch.as_tensor(real_id, dtype=torch.long, device=logits.device).sort().values
    pseudo_ood = torch.as_tensor(pseudo_ood, dtype=torch.long, device=logits.device).sort().values
    cls = F.cross_entropy(logits[real_id], targets[real_id])
    energy = -torch.logsumexp(logits, dim=1)
    inside, outside = energy[real_id], energy[pseudo_ood]
    n = min(len(inside), len(outside))
    # Preserve the source implementation's equal-length truncation, not an
    # unannounced switch to separate means over every ID training node.
    reg = (F.relu(inside[:n] - m_in).square() + F.relu(m_out - outside[:n]).square()).mean()
    return {"classification": cls, "exposure": reg, "total": cls + lam * reg}


def propagation_matrix(edge_index, n):
    row, col = edge_index.cpu()
    degree = torch.bincount(col, minlength=n).float()
    values = degree[col].reciprocal()
    return torch.sparse_coo_tensor(torch.stack([col, row]), values, (n, n)).coalesce()


def gnnsafe_energy(logits, matrix, steps=8, alpha=0.):
    # The reference propagates +logsumexp as ID confidence. Negation makes
    # OOD positive here, consistently for AUROC/AP-OOD/FPR95.
    confidence = torch.logsumexp(logits, dim=1).unsqueeze(1)
    for _ in range(steps):
        confidence = alpha * confidence + (1 - alpha) * torch.sparse.mm(matrix, confidence)
    return -confidence.squeeze(1)
