"""Part B rule - Cross-subject evaluation with a fixed, documented subject-wise fold scheme.

Brief: "Cross-subject evaluation only: no trial from a test subject may touch training, normalisation statistics,
channel selection or early stopping. Use a fixed, documented subject-wise fold scheme on the subjects and trials kept
in Part A."

Live part: rebuilds the folds with task1/train.py and checks, for every test fold and every training-set size and seed
used in B2, that test, validation (early stopping) and training subjects are disjoint. Then checks the saved runs.
Run: python task1/asks/B0_cross_subject_folds.py
"""
import inspect
from _common import *

header("Part B  Subject-wise folds, nothing from a test subject in training",
       "cross-subject only; no test-subject trial in training, normalisation, channel selection or early stopping; "
       "fixed documented subject-wise folds on the subjects and trials kept in Part A.",
       "8 folds of 5 subjects (same as A3). Test = fold f, validation (early stopping only) = fold f+1, training = other 30.")
print_decision("D22")

train = pipeline("train")
fo = train.folds()
section("The fold of every subject (np.random.default_rng(0).permutation of subjects 70-109, fold = rank mod 8)")
F = pd.DataFrame(sorted(fo.items()), columns=["subject", "fold"])
for f in range(train.N_FOLDS):
    print(f"  fold {f}: subjects {F[F.fold == f].subject.tolist()}")
saved = csv("a3_folds.csv")
print(f"  same assignment as A3 (outputs/a3_folds.csv): {saved.merge(F, on='subject', suffixes=('_a3', '')).eval('fold_a3 == fold').all()}")

section("Disjointness for every split used in B2 (test folds 0-3, sizes 5/10/20/30, seeds 0-2)")
bad = 0
for f in range(4):
    for n in (5, 10, 20, 30):
        for seed in (0, 1, 2):
            tr, va, te = train.split(f, n, seed)
            bad += bool(set(tr) & set(te) or set(va) & set(te) or set(tr) & set(va)) or len(tr) != n
    tr, va, te = train.split(f, 30, 0)
    print(f"  test fold {f}: test {te}  validation {va}  training {len(tr)} subjects")
print(f"  splits with any overlap or wrong size: {bad} of 48")

section("Where each forbidden use is prevented (task1/train.py source)")
src = inspect.getsource(train.fit).splitlines()
for line in src:
    if any(k in line for k in ("Norm().fit", "early stopping", "split(test_fold", "transform(Xtr")):
        print("  " + line.strip())
say("Normalisation: per-channel mean/std fitted on training subjects only, applied unchanged to validation and test. "
    "Early stopping: validation-subject loss. Channel selection (confound-removed run): fixed electrode geometry "
    "(task1/confound_free.py), no statistic from any subject. Trials: the Part A cache (20 dropped trials, A1).")

R = csv("partb/b2_all_runs.csv")
section("Saved runs: every one of the 156 B2 runs used test folds 0-3 and trained on the stated number of subjects")
print(f"  runs: {len(R)}; test folds used: {sorted(map(int, R.fold.unique()))}; training sizes: {sorted(map(int, R.n_train.unique()))}")
n_test = R.groupby("fold").n_test.first()
print(f"  test trials per fold: {n_test.to_dict()} -> {n_test.sum()} test trials in total (same 20 subjects everywhere)")
print_decision("D24")
finish()
