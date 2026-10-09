"""Defect 1 evidence: the given split shuffles RECORDINGS, so a subject's two nights can land on both sides.
Re-runs the given split rule (same SEED, same 20%) and the fixed subject-level rule, and counts shared subjects."""
import random, sys
sys.path.insert(0, "..")
from sleep_pipeline import get_files, subject_of, SUBJECTS

files = get_files(SUBJECTS)
print(f"{len(files)} recordings from {len({subject_of(f) for f in files})} subjects")
for seed in (42, 43, 44):
    f = list(files); random.Random(seed).shuffle(f)                 # given rule
    n_test = max(1, int(0.2 * len(f)))
    test_s = {subject_of(x) for x in f[:n_test]}; train_s = {subject_of(x) for x in f[n_test:]}
    shared = sorted(test_s & train_s)
    subj = sorted({subject_of(x) for x in files}); random.Random(seed).shuffle(subj)   # fixed rule
    n = max(1, int(0.2 * len(subj)))
    fixed_shared = set(subj[:n]) & set(subj[2 * n:])
    print(f"seed {seed}: given split  test recordings {n_test}, test subjects {len(test_s)}, "
          f"also in train {len(shared)} {shared}  |  fixed split shared subjects {len(fixed_shared)}")
