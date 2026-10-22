import numpy as np

def partition(rows, labels, energy, val, corrections, ood_quantile):
    assert ood_quantile in (.8, .9, .95)
    rows = np.asarray(rows, dtype=np.int64); val = np.asarray(val, dtype=np.int64)
    assert np.isfinite(energy[np.r_[rows, val]]).all()
    q = np.quantile(energy[val].astype(float), [.25, .5, ood_quantile, .99], method='linear')
    assert q[0] <= q[1] < q[2] <= q[3], 'Teacher thresholds are degenerate'
    e = energy[rows]; llm = labels[rows]
    inside = (llm == 0)&(e <= q[1])
    outside = (llm == 1)&(e >= q[2])
    if corrections:
        inside |= (llm == 1)&(e <= q[0])
        outside |= (llm == 0)&(e >= q[3])
    assert not (inside&outside).any()
    return {'IND': rows[inside].tolist(), 'OOD': rows[outside].tolist(),
            'PENDING': rows[~(inside|outside)].tolist(),
            'thresholds': dict(zip(('q25','q50','q_ood','q99'), map(float, q)))}
