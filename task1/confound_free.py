"""B2 'confound removed' input: local Laplacian over all 64 electrodes, then only the 21 sensorimotor electrodes.
A3 found the dominant non-motor route is the eye-movement signal (frontal 0-0.5 s, 72% pooled; frontal 0.5-4 s, 75%).
Dropping frontal channels alone is not enough because the eye field spreads over the scalp by volume conduction (A3 N2,
everything except sensorimotor, 70.6%); the Laplacian subtracts what a site shares with its 4 neighbours, so a spatially
broad far field cancels while focal mu/beta ERD under C3/C4 survives (D4, DECISIONS D24)."""
import numpy as np
import data, erd, confound

KEEP = confound.SENSORIMOTOR
_W = None


def weights():
    global _W
    if _W is None:
        names = list(data.load_subject(data.SUBJECTS[0], causal=True)["ch_names"])
        W = np.eye(len(names))
        for c in names:
            i = names.index(c)
            W[i, erd.neighbours(names, c)] -= 0.25
        _W = W[confound.idx(names, KEEP)].astype(np.float32)   # (21, 64): fixed geometry, no data-fitted statistic
    return _W


def transform(X, subjects=None):
    return np.einsum("kc,nct->nkt", weights(), X)
