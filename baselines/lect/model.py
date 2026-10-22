"""LECT equations 9/10/12 and the approved full model."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class LECT(nn.Module):
    def __init__(self, input_dim, classes, cfg):
        super().__init__()
        self.project = nn.Linear(input_dim, cfg["project_dim"])
        self.conv1 = GCNConv(cfg["project_dim"], cfg["hidden_dim"])
        self.bn = nn.BatchNorm1d(cfg["hidden_dim"])
        self.conv2 = GCNConv(cfg["hidden_dim"], classes)
        self.dropout = cfg["dropout"]

    def forward(self, x, s):
        h = F.relu(self.project(x))
        h = torch.sparse.mm(s, self.conv1.lin(h)) + self.conv1.bias
        h = F.dropout(F.relu(self.bn(h)), self.dropout, self.training)
        return torch.sparse.mm(s, self.conv2.lin(h)) + self.conv2.bias


def losses(z, labels, real, pairs, triples, cfg):
    energy = -torch.logsumexp(z, dim=1)
    cls = F.cross_entropy(z[real], labels[real])
    if pairs is None:
        pair = triplet = energy.new_zeros(())
    else:
        if len(pairs) == 0 or len(triples) == 0:
            raise ValueError("LECT needs nonempty linked pairs and triples")
        inside, outside = pairs.T
        positive, center, negative = triples.T
        pair = F.relu(cfg["gamma"] - (energy[outside] - energy[inside])).mean()
        triplet = F.relu((energy[positive] - energy[center]).abs() -
                         (energy[negative] - energy[center])).mean()
    return {"classification": cls, "pair": pair, "triplet": triplet,
            "total": cls + cfg["lambda_pair"] * pair + cfg["lambda_triplet"] * triplet}


def generation_plan(real, n, seed, cfg):
    rng = np.random.default_rng(seed)
    near = int(cfg["generated_per_seed"] * cfg["near_fraction"])
    regimes = rng.permutation(["near"] * near + ["far"] * (cfg["generated_per_seed"] - near))
    return [{"node_id": f"lect_s{seed}_{i:04d}", "row": n + i, "regime": str(regime),
             "neighbors": rng.choice(real, int(rng.integers(1, cfg["max_neighbors"] + 1)), replace=False).tolist()}
            for i, regime in enumerate(regimes)]


def supervision(real, edges, nodes, n):
    real = set(map(int, real))
    neighbors = {row: [] for row in real}
    for a, b in edges.T:
        if int(a) in real and int(b) in real:
            neighbors[int(a)].append(int(b))
    pairs, triples = [], []
    for i, node in enumerate(nodes):
        row = n + i
        if not 1 <= len(node["neighbors"]) <= 6 or not set(node["neighbors"]) <= real:
            raise ValueError("Synthetic neighbors must be real training ID nodes")
        if len(node["neighbors"]) != len(set(node["neighbors"])):
            raise ValueError("Duplicate synthetic edge")
        for center in node["neighbors"]:
            pairs.append((center, row))
            triples.extend((pos, center, row) for pos in neighbors[center])
    if not pairs or not triples:
        raise ValueError("No legal LECT supervision")
    return np.asarray(pairs, dtype=np.int64), np.asarray(triples, dtype=np.int64)


FIELDS = {"main_domain", "ood_label", "relation", "title", "description"}


def validate_sample(value, unused=None):
    if not isinstance(value, dict) or set(value) != FIELDS:
        raise ValueError("Unexpected generated sample fields")
    if any(not isinstance(v, str) or not v.strip() or len(v) > 6000 for v in value.values()):
        raise ValueError("Invalid generated text")
    # Validate format only. The 80-160 word target is a prompt request, not an
    # additional semantic quality filter or an assertion of OOD correctness.
    return {k: v.strip() for k, v in value.items()}
