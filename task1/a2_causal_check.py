"""A2 re-check on the minimum-phase (causal) pre-filter (DECISIONS D14): does the zero-phase filter change the lateralisation labels?
Laplacian reference only (the A2 primary). Compares with outputs/a2_lateralisation.csv.   python task1/a2_causal_check.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, str(Path(__file__).parent))
import data, erd

def main():
    rows = Parallel(n_jobs=-1)(delayed(erd.analyse)(s, "laplacian", False, True) for s in data.SUBJECTS)
    new = pd.DataFrame(rows).drop(columns="ref")
    new.to_csv(data.ROOT / "outputs" / "a2_lateralisation_causal.csv", index=False)
    old = pd.read_csv(data.ROOT / "outputs" / "a2_lateralisation.csv")
    m = old.merge(new, on="subject", suffixes=("_zero", "_causal"))
    print("present (zero-phase):", (m.label_zero == "present").sum(), "| present (causal):", (m.label_causal == "present").sum())
    print("labels identical for", (m.label_zero == m.label_causal).sum(), "of", len(m), "subjects; changed:",
          m[m.label_zero != m.label_causal].subject.tolist())
    for b in ("mu", "beta"):
        print(f"{b}: corr of LI = {np.corrcoef(m[f'{b}_LI_zero'], m[f'{b}_LI_causal'])[0,1]:.3f}, "
              f"corr of contra ERD dB = {np.corrcoef(m[f'{b}_contra_db_zero'], m[f'{b}_contra_db_causal'])[0,1]:.3f}, "
              f"max |p diff| = {np.abs(m[f'{b}_p_zero'] - m[f'{b}_p_causal']).max():.4f}")
if __name__ == "__main__":
    main()
