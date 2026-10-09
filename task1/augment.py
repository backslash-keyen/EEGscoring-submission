"""B2 intervention: train with random simulated cap displacements (DECISIONS D25).
Each training trial is replaced, with probability 1, by M_k x for a matrix M_k drawn from a fixed bank of displacements
of 0-15 mm in uniformly random directions; validation and test data are not augmented."""
import numpy as np
import data, spatial

MAX_MM, N_BANK = 15.0, 256


class Augmenter:
    def __init__(self, seed):
        names = list(data.load_subject(data.SUBJECTS[0], causal=True)["ch_names"])
        rng = np.random.default_rng(777 + seed)
        ang = rng.uniform(0, 2 * np.pi, N_BANK)
        mm = MAX_MM * np.sqrt(rng.uniform(0, 1, N_BANK))   # uniform over the disc of radius 15 mm
        self.bank = np.stack([spatial.displacement_matrix(names, s, (np.cos(a), np.sin(a))) for s, a in zip(mm, ang)]).astype(np.float32)
        self.rng = rng

    def __call__(self, X):
        k = self.rng.integers(0, N_BANK, len(X))
        return np.einsum("nkc,nct->nkt", self.bank[k], X)
