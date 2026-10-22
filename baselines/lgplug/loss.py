"""Parameter-independent LG-Plug objectives from paper equations 3-6 and 15."""
import torch
import torch.nn.functional as F


def alignment_loss(node_z, node_h, edge_z, edge_h, log_scale, edge_weight):
    """Node pairs are [N,D]; edge endpoints are [E,2,D]."""
    if node_z.ndim != 2 or node_z.shape != node_h.shape or len(node_z) < 2:
        raise ValueError("Alignment needs at least two matching node-text pairs")
    if edge_z.ndim != 3 or edge_z.shape != edge_h.shape or edge_z.shape[1] != 2 or not len(edge_z):
        raise ValueError("Alignment needs matching graph-text edge endpoints")
    z, h = F.normalize(node_z, dim=1), F.normalize(node_h, dim=1)
    logits = (z @ h.T) * log_scale.exp()
    target = torch.arange(len(z), device=z.device)
    node = .5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))
    ez, eh = F.normalize(edge_z, dim=-1), F.normalize(edge_h, dim=-1)
    graph_similarity = (ez[:, 0] * ez[:, 1]).sum(-1)
    text_similarity = (eh[:, 0] * eh[:, 1]).sum(-1)
    edge = (graph_similarity - text_similarity).square().mean()
    return {"node": node, "edge": edge, "total": node + edge_weight * edge}


def exposure_loss(scores, real_id, pseudo_ood, id_margin, ood_margin):
    """Separate population means, retaining gradients through the OOD scorer."""
    if scores.ndim != 1 or not len(real_id) or not len(pseudo_ood):
        raise ValueError("Exposure needs nonempty real ID and pseudo OOD groups")
    inside = F.relu(scores[real_id] - id_margin).square().mean()
    outside = F.relu(ood_margin - scores[pseudo_ood]).square().mean()
    return {"ID": inside, "OOD": outside, "total": inside + outside}
