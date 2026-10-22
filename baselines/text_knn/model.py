import numpy as np

def normalize_rows(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Expected finite representation matrix")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Cosine distance is undefined for zero vectors")
    return x / norms

def knn_distance(query, reference, k):
    if not isinstance(k, int) or not 1 <= k <= len(reference):
        raise ValueError("Invalid neighbor count")
    q, r = normalize_rows(query), normalize_rows(reference)
    distance = np.clip(1 - q @ r.T, 0, 2)
    return np.partition(distance, k - 1, axis=1)[:, k - 1]
