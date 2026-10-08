# %% [markdown]
# # A3, step 3: every route, tested with a restricted decoder
# Each decoder is shown only part of the signal: some channels, one band, or one time window. If it beats chance, that part of the
# signal carries the cue side without any motor imagery. Numbers below are read from the saved run (python task1/confound.py).

# %%
from common import *
summ = pd.read_csv(OUT / "a3_summary.csv").set_index("decoder")
within = pd.read_csv(OUT / "a3_within_subject.csv")
ROUTES = [("F1", "eye movement: frontal, 0-0.5 s"), ("F2", "eye movement: frontal, 0.5-4 s"), ("O1", "visual: occipital, 0-0.5 s"),
          ("O2", "visual/attention: occipital mu+beta"), ("G1", "muscle: temporal 30-40 Hz (weak test)"),
          ("N2", "all but sensorimotor, 0-0.5 s"), ("M1", "MOTOR strip mu+beta (the real signal)"), ("A1", "all channels mu+beta"),
          ("R2", "previous trial label (no EEG)")]
tab = pd.DataFrame({"what the decoder sees": [d for _, d in ROUTES],
                    "within-subject mean": [summ.loc[k, "mean_acc"] for k, _ in ROUTES],
                    "subjects above threshold /40": [int(summ.loc[k, "n_above"]) for k, _ in ROUTES],
                    "pooled cross-subject": [summ.loc[k, "pooled_xs_acc"] for k, _ in ROUTES],
                    "pre-cue twin (pooled)": [summ.pooled_xs_acc.get(k + "_pre", np.nan) for k, _ in ROUTES]}, index=[k for k, _ in ROUTES])
print(tab.round(3).to_string())
print(f"\npooled threshold (n = 1793): {C.binom_thr(1793):.3f}; within-subject thresholds: {within.thr_binom.min():.3f}-{within.thr_binom.max():.3f}")

# %%
order = [k for k, _ in ROUTES]
fig, ax = plt.subplots(figsize=(11, 4.2))
for i, k in enumerate(order):
    a = within[within.decoder == k].acc.values
    ax.scatter(i + np.random.default_rng(i).uniform(-.2, .2, len(a)), a, s=14, alpha=.55)
    ax.hlines(a.mean(), i - .32, i + .32, color="k")
ax.axhline(0.5, color="gray", lw=.8)
ax.fill_between([-0.5, len(order) - 0.5], within.thr_binom.min(), within.thr_binom.max(), color="r", alpha=.12, label="per-subject 5% threshold range")
ax.set_xticks(range(len(order))); ax.set_xticklabels(order); ax.set_ylabel("leave-one-run-out accuracy"); ax.legend()
ax.set_title("one dot per subject, black bar = mean")
save(fig, "03_within_subject.png")

# %% [markdown]
# ## Pooled across subjects (train on 35 subjects, test on 5 unseen), against the pre-cue twin
# The pre-cue twin sees the same channels and features but a window BEFORE the cue. No label information can exist there, so it must sit at chance.

# %%
pooled = pd.read_csv(OUT / "a3_cross_subject.csv").set_index("decoder")
ids = ["F1", "F2", "O1", "O2", "N2", "M1", "A1", "R2"]
fig, ax = plt.subplots(figsize=(9, 3.8)); w = 0.38
ax.bar(np.arange(len(ids)) - w / 2, [pooled.loc[k, "acc"] for k in ids], w, label="after the cue")
ax.bar(np.arange(len(ids)) + w / 2, [pooled.loc[k + "_pre", "acc"] if k + "_pre" in pooled.index else 0 for k in ids], w, label="pre-cue twin", color="tab:gray")
ax.axhline(0.5, color="k", lw=.8); ax.axhline(C.binom_thr(1793), color="r", ls=":", label="5% threshold")
ax.set_ylim(0.4, 0.8); ax.set_xticks(range(len(ids))); ax.set_xticklabels(ids); ax.set_ylabel("pooled accuracy"); ax.legend(ncol=3)
save(fig, "03_pooled.png")

# %% [markdown]
# ## Prediction scorecard (predictions were committed in 3bdff2a before any result)

# %%
score = pd.DataFrame([
    ("1 routes", "eyes + visual", "eyes strongly, visual weakly, plus trial history", "partly right"),
    ("2 frontal subjects", "5-10 of 40", f"{int(summ.loc['F1', 'n_above'])} of 40", "wrong"),
    ("3 frontal pooled", "50-52%", f"{pooled.loc['F1', 'acc'] * 100:.1f}%", "wrong"),
    ("4 occipital pooled", "53-57%", f"{pooled.loc['O1', 'acc'] * 100:.1f}%", "slightly above"),
    ("5 pre-cue / order", "at chance", f"previous label alone {pooled.loc['R2', 'acc'] * 100:.1f}%", "wrong"),
    ("6 motor pooled", "56-62%", f"{pooled.loc['M1', 'acc'] * 100:.1f}%", "slightly below")], columns=["#", "predicted", "found", "verdict"])
print(score.to_string(index=False))
