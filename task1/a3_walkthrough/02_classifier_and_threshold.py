# %% [markdown]
# # A3, step 2: the classifier and what counts as "above chance"
# One classifier everywhere: L2 logistic regression, C = 0.1 (fixed in advance, never tuned), features standardised on the training data only.

# %%
from common import *
from scipy.stats import binom
D = data.load_subject(SUBJECT, causal=True)
X, y, run = D["X"].astype(float), D["y"], D["run"]
names = list(D["ch_names"])
spec = C.decoder_specs(names)

def loro_by_hand(F, y, run):
    """The same thing C.loro_acc does, written out so every step is visible."""
    pred = np.empty(len(y), int)
    for r in np.unique(run):
        test = run == r
        mu, sd = F[~test].mean(0), F[~test].std(0) + 1e-12            # scaler statistics from TRAINING trials only
        clf = C.LogisticRegression(C=C.C_REG, max_iter=500).fit((F[~test] - mu) / sd, y[~test])
        pred[test] = clf.predict((F[test] - mu) / sd)
    return (pred == y).mean()

for k in ("F1", "O1", "M1", "F1_pre"):
    F = spec[k][2](X)
    print(f"{k:7s} {spec[k][1]:42s} features {F.shape[1]:3d}  accuracy {loro_by_hand(F, y, run):.3f}  (library check: {C.loro_acc(F, y, run):.3f})")

# %% [markdown]
# ## Chance threshold: exact binomial
# With n test trials and NO information, the number correct is Binomial(n, 0.5). The threshold is the smallest accuracy k/n with P(X >= k) <= 5%.
# Small n means a high bar: 50% is not chance-level evidence, ~64% is.

# %%
n = len(y); thr = C.binom_thr(n); k_thr = round(thr * n)
print(f"n = {n} trials -> threshold {thr:.3f} ({k_thr}/{n} correct); P(X >= {k_thr}) = {binom.sf(k_thr-1, n, .5):.4f}, P(X >= {k_thr-1}) = {binom.sf(k_thr-2, n, .5):.4f}")
for nn in (36, 45, 424, 1249, 1793):
    print(f"  n = {nn:5d}: threshold {C.binom_thr(nn):.3f}")
fig, ax = plt.subplots(figsize=(7, 3))
ks = np.arange(n + 1); ax.bar(ks / n, binom.pmf(ks, n, .5), width=.8 / n, color="lightgray")
ax.axvline(thr, color="r", label=f"5% threshold {thr:.2f}"); ax.set_xlabel("accuracy under no information"); ax.legend()
save(fig, "02_binomial_null.png")

# %% [markdown]
# ## Check the binomial against a permutation null
# CV predictions are correlated, so the binomial is only approximate. Permute labels within each run, redo the whole cross-validation,
# and see how often the null beats the threshold (it should be ~5%).

# %%
F = spec["F1_pre"][2](X)      # a decoder that SHOULD be at chance: it only sees the rest period before the cue
rng = np.random.default_rng(0)
null = np.array([C.loro_acc(F, C.perm_labels(y, run, rng), run) for _ in range(300)])
print(f"permutation null of F1_pre: mean {null.mean():.3f}, sd {null.std():.3f}; fraction above binomial threshold {(null >= thr).mean():.3f} (nominal 0.05)")
allp = pd.read_csv(OUT / "a3_summary.csv").perm_false_pos
print(f"across all 20 decoders x 40 subjects the false-positive rate was {allp.min():.3f} to {allp.max():.3f}")
fig, ax = plt.subplots(figsize=(7, 3))
ax.hist(null, bins=20, color="lightgray"); ax.axvline(thr, color="r"); ax.set_xlabel("accuracy with labels permuted (F1_pre)")
save(fig, "02_permutation_null.png")
