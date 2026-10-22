import numpy as np
import torch
from dataclasses import dataclass
from src.numerics import GraphData

@dataclass(slots=True)
class TrainingView:
    graph: GraphData
    roles: dict[str, np.ndarray]
    cls_targets: torch.Tensor
    id_definitions: list[dict]

def make_splits(labels, cfg):
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.dtype.kind not in "iu" or labels.min() < 0:
        raise ValueError("Expected one integer class per node")
    classes = np.unique(labels)
    if cfg["num_id_classes"] >= len(classes):
        raise ValueError("Class split must leave OOD classes")
    order = np.random.default_rng(cfg["class_split_seed"]).permutation(classes)
    known = order[:cfg["num_id_classes"]].tolist()
    rng = np.random.default_rng(cfg["seeds"][0])
    train, val = [], []
    a, b = cfg["id_train_per_class"], cfg["id_val_per_class"]
    for c in known:
        nodes = rng.permutation(np.flatnonzero(labels == c))
        if len(nodes) < a + b + 10:
            raise ValueError("ID class has fewer than ten remaining unlabeled nodes")
        train.extend(nodes[:a].tolist())
        val.extend(nodes[a:a + b].tolist())
    unlabeled = np.setdiff1d(np.arange(len(labels)), train + val)
    n_candidate = min(cfg["candidate_budget"], int(np.floor(cfg["candidate_fraction_cap"] * len(unlabeled))))
    candidate = rng.choice(unlabeled, n_candidate, replace=False)
    folds = np.random.default_rng(cfg["fold_seed"]).permutation(candidate)
    midpoint = (len(folds) + 1) // 2
    roles = {"L": np.asarray(train, dtype=np.int64), "V": np.asarray(val, dtype=np.int64),
             "A": folds[:midpoint], "B": folds[midpoint:], "Q": np.setdiff1d(unlabeled, candidate)}
    validate_roles(roles, len(labels))
    targets = torch.full((len(labels),), -100, dtype=torch.int64)
    mapping = {c: i for i, c in enumerate(known)}
    visible = np.concatenate([roles["L"], roles["V"]])
    targets[visible] = torch.tensor([mapping[int(c)] for c in labels[visible]])
    plan = {"permuted_classes": order.tolist(), "id_classes": known,
            "ood_classes": sorted(set(classes.tolist()) - set(known)), "id_class_map": mapping,
            "rng": "numpy.default_rng / PCG64", "node_split_seed": cfg["seeds"][0],
            "fold_seed": cfg["fold_seed"], "candidate_sampling": "uniform without replacement over all U, no class balancing"}
    return roles, targets, plan

def validate_roles(roles, n):
    if set(roles) != set(ROLES) or any(a.ndim != 1 or a.dtype != np.int64 for a in roles.values()):
        raise ValueError("Unexpected role schema")
    all_rows = np.concatenate(list(roles.values()))
    if not np.array_equal(np.sort(all_rows), np.arange(n)):
        raise ValueError("L/V/A/B/Q must be disjoint and cover all nodes")
    if len(roles["A"]) not in (len(roles["B"]), len(roles["B"]) + 1):
        raise ValueError("A/B must be equally split, with odd surplus in A")

def build_training_view(graph, roles, targets, definitions):
    validate_roles(roles, len(graph.node_ids))
    k = len(definitions)
    if targets.dtype != torch.int64 or targets.shape != (len(graph.node_ids),):
        raise ValueError("Invalid targets")
    visible = np.concatenate([roles["L"], roles["V"]])
    hidden = np.concatenate([roles[r] for r in ("A", "B", "Q")])
    if (targets[hidden] != -100).any() or (targets[visible] < 0).any() or (targets[visible] >= k).any():
        raise ValueError("Class targets leak outside L/V")
    if [d["id"] for d in definitions] != [str(i) for i in range(k)]:
        raise ValueError("ID definition indices do not match the classification head")
    return TrainingView(graph, {r: v.copy() for r, v in roles.items()}, targets.clone(), definitions)

ROLES = ("L", "V", "A", "B", "Q")
