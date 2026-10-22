"""Fixed candidate sampling and dynamic semantic-codebook rules for LG-Plug."""
import math
import unicodedata

import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits


def key(label):
    return unicodedata.normalize("NFKC", label).strip().casefold()


def validate_label(value, max_chars):
    if not isinstance(value, dict) or set(value) != {"label"} or not isinstance(value["label"], str):
        raise ValueError("Expected exactly one JSON string field named label")
    label = unicodedata.normalize("NFKC", value["label"]).strip()
    label.encode("utf-8")
    if not label or len(label) > max_chars or any(c in label for c in ("\n", "\r")):
        raise ValueError("Invalid category name length or line breaks")
    if key(label) in {"ood", "id", "other", "others", "unknown", "none", "uncertain", "unclassified", "miscellaneous", "out-of-distribution"}:
        raise ValueError("A concrete category name is required")
    return label


class Codebook:
    def __init__(self, names):
        if len({key(name) for name in names}) != len(names):
            raise ValueError("ID category names collide after normalization")
        self.names = list(names)
        self.counts = [0] * len(names)
        self.id_count = len(names)

    def add(self, label):
        keys = [key(name) for name in self.names]
        if key(label) in keys:
            index = keys.index(key(label))
        else:
            index = len(self.names)
            self.names.append(label)
            self.counts.append(0)
        self.counts[index] += 1
        return index

    def state(self):
        return [{"name": name, "count": count, "is_ID": i < self.id_count}
                for i, (name, count) in enumerate(zip(self.names, self.counts))]

    def top(self, count):
        positive = [i for i, n in enumerate(self.counts) if n > 0]
        return sorted(positive, key=lambda i: (-self.counts[i], i))[:count]

    def exposure(self, records, count):
        top = self.top(count)
        if not top or any(i < self.id_count for i in top):
            return []
        return sorted(r["row"] for r in records if r["status"] == "valid" and r["category"] in top)


def consensus(records):
    return bool(records) and all(r["status"] == "valid" for r in records) and len({r["category"] for r in records}) == 1


def candidate_clusters(rows, z, ids, seed, cfg):
    rows = np.sort(np.asarray(rows, dtype=np.int64))
    if len(set(rows)) != len(rows) or len(rows) < cfg["clusters"] or not np.isfinite(z[rows]).all():
        raise ValueError("Invalid LG-Plug candidate rows or representations")
    with threadpool_limits(limits=1):
        model = KMeans(n_clusters=cfg["clusters"], n_init=cfg["kmeans_n_init"], max_iter=cfg["kmeans_max_iter"],
                       tol=cfg["kmeans_tol"], random_state=seed, algorithm="lloyd").fit(z[rows])
    if len(np.unique(model.labels_)) != cfg["clusters"]:
        raise ValueError("Fewer nonempty clusters than the fixed protocol requires")
    groups = []
    for label in range(cfg["clusters"]):
        members = rows[model.labels_ == label]
        distances = ((z[members] - model.cluster_centers_[label]) ** 2).sum(1)
        order = sorted(range(len(members)), key=lambda i: (float(distances[i]), ids[members[i]]))
        representatives = members[order[:math.ceil(cfg["rho"] * len(members))]]
        groups.append({"members": members.tolist(), "representatives": representatives.tolist(),
                       "center": model.cluster_centers_[label].tolist()})
    groups.sort(key=lambda group: min(ids[row] for row in group["members"]))
    rng = np.random.default_rng(seed)
    for index, group in enumerate(groups):
        order = rng.permutation(group["representatives"]).tolist()
        order = order[:cfg["batch_size"] * cfg["trials"]]
        group.update(cluster=index, batches=[order[start:start + cfg["batch_size"]]
                                            for start in range(0, len(order), cfg["batch_size"])])
    return groups
