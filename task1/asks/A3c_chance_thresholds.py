"""A3 (part 3) - Chance-level threshold for every decoding accuracy, and how it was derived.

Brief: "For every decoding accuracy, give the chance-level threshold for that number of trials and how you derived it."

Derivation: with n test trials and a classifier that has no information, the number correct is X ~ Binomial(n, 0.5).
The threshold is the smallest accuracy k/n with P(X >= k) <= 0.05 (exact, one-sided). Checked two ways below:
simulation of an information-free classifier, and the label-permutation null saved by the pipeline (whole CV repeated).
Change N to see the threshold for any trial count.
Run: python task1/asks/A3c_chance_thresholds.py
"""
from scipy.stats import binom
from _common import *

N = 43              # trials of one subject (S072); try 36 (S100) or 1793 (all subjects pooled)
ALPHA = 0.05

header("A3.3  Chance thresholds",
       "for every decoding accuracy, the chance-level threshold for that number of trials and how it was derived.",
       "exact one-sided binomial: smallest k/n with P(X >= k | n, p = 0.5) <= 0.05; validated by simulation and by "
       "the label-permutation null of the actual cross-validation.")
print_decision("D15")
confound = pipeline("confound")

section(f"Derivation for n = {N}, step by step")
for k in range(int(N * 0.5), N + 1):
    p = binom.sf(k - 1, N, 0.5)             # P(X >= k)
    flag = "  <- first k with P <= 0.05: threshold" if p <= ALPHA and binom.sf(k - 2, N, 0.5) > ALPHA else ""
    print(f"  k = {k:3d}  accuracy {k / N:.3f}  P(X >= k) = {p:.4f}{flag}")
    if p < 0.001:
        break
thr = confound.binom_thr(N)
print(f"\n  pipeline function confound.binom_thr({N}) = {thr:.4f}")

section("Check 1: simulate 200000 information-free classifiers on n trials")
rng = np.random.default_rng(0)
acc = rng.binomial(N, 0.5, 200_000) / N
print(f"  fraction of random classifiers reaching the threshold: {(acc >= thr).mean():.4f} (must be <= 0.05; "
      f"below 0.05 because accuracy moves in steps of 1/{N})")

section("Check 2: label-permutation null of the real cross-validation (saved by task1/confound.py)")
S = csv("a3_summary.csv")
table(S[["decoder", "perm_false_pos"]].assign(nominal=0.05))
say("perm_false_pos = fraction of label-permuted re-runs (200 per subject and decoder) that cleared the binomial "
    "threshold. 0.03-0.08 against a nominal 0.05: the binomial threshold is slightly liberal for cross-validated "
    "accuracy, so single-subject calls are reported uncorrected and the group claim uses the excess test (A3b).")

section("Every trial count used in Task 1 and its threshold")
aud = csv("audit.csv")
ns = sorted(set((aud.n_left_used + aud.n_right_used).tolist()))
rows = [dict(where=f"within subject (n = trials kept for that subject)", n=n, threshold=confound.binom_thr(n)) for n in ns]
fx = [("A3 pooled over 40 subjects", 1793), ("A3 follow-up: repeat trials", 424), ("A3 follow-up: alternating trials", 1249),
      ("A3 follow-up: trials with a previous label", 1673), ("B2 test set (folds 0-3, 20 subjects)", 913)]
rows += [dict(where=w, n=n, threshold=confound.binom_thr(n)) for w, n in fx]
T = pd.DataFrame(rows)
table(T)
saved = csv("a3_within_subject.csv").groupby("n").thr_binom.first()
print(f"\n  per-subject thresholds equal the ones stored with every accuracy in a3_within_subject.csv: "
      f"{all(np.isclose(saved.loc[n], confound.binom_thr(n)) for n in saved.index)}")
say("For the balanced repeat/alternate accuracy (mean of two proportions) no exact binomial exists; there the "
    "threshold is the normal approximation 0.5 + 1.645 * 0.5 * sqrt(0.25/n_rep + 0.25/n_alt) = 0.523 (task1/a3_thresholds.py).")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
n_grid = np.unique(np.geomspace(10, 2000, 200).astype(int))
ax[0].plot(n_grid, [confound.binom_thr(n) for n in n_grid], color="k")
ax[0].scatter(T.n, T.threshold, color="C3", zorder=3)
ax[0].set_xscale("log"); ax[0].axhline(.5, color="0.6", lw=.6)
ax[0].set_xlabel("number of test trials n"); ax[0].set_ylabel("5% chance threshold (accuracy)")
ax[0].set_title("threshold falls with n (red = counts used here)", fontsize=9)
k = np.arange(N + 1)
ax[1].bar(k / N, binom.pmf(k, N, .5), width=.8 / N, color=np.where(k / N >= thr, "C3", "0.6"))
ax[1].set_xlabel("accuracy of a classifier with no information"); ax[1].set_title(
    f"n = {N}: P(accuracy >= {thr:.3f}) = {binom.sf(round(thr * N) - 1, N, .5):.3f}", fontsize=9)
save(fig, "A3c_chance_thresholds.png")
finish()
