# %% [markdown]
# # A3, step 1: what goes into the classifier, and how the data is split
# The task: tell imagined LEFT fist from imagined RIGHT fist. A3 asks what ELSE in the recording separates left from right trials.
# The target appears on the left or right of the screen, so anything driven by the target side (eyes, vision) is a confound.

# %%
from common import *
D = data.load_subject(SUBJECT, causal=True)       # causal = minimum-phase 1-40 Hz filter (nothing leaks backwards in time)
X, y, run = D["X"].astype(float), D["y"], D["run"]
names = list(D["ch_names"])
print(f"S{SUBJECT}: X shape {X.shape} = (trials, channels, samples); window -1.5..4.0 s around the cue at 160 Hz")
print("trials per run:", {int(r): int((run == r).sum()) for r in np.unique(run)}, "| left:", int((y == 0).sum()), "right:", int((y == 1).sum()))

# %% [markdown]
# ## What a decoder sees: one flat row of numbers per trial
# Example, decoder F1 ("frontal, first 0.5 s"): 12 frontal channels, each baseline-corrected on its own pre-cue mean, averaged in 100 ms bins.

# %%
spec = C.decoder_specs(names)
F1 = spec["F1"][2](X)
print("F1 feature matrix:", F1.shape, "= (trials, 12 channels x 5 bins)")
cols = [f"{c}@{int(b*1000)}-{int(b*1000)+100}ms" for b in np.arange(0, 0.5, 0.1) for c in C.FRONT]
print(pd.DataFrame(F1[:3, :6], columns=cols[:6]).round(1).to_string())
fig, ax = plt.subplots(1, 3, figsize=(14, 3.4))
t = C.TIMES
for k, ch in enumerate(["F7", "F8"]):
    i = C.idx(names, [ch])[0]
    for lab, col, nm in ((0, "tab:blue", "left cue"), (1, "tab:red", "right cue")):
        base = X[y == lab][:, i][:, (t >= -0.5) & (t < 0)].mean(1, keepdims=True)
        ax[k].plot(t, ((X[y == lab][:, i] - base) * 1e6).mean(0), color=col, label=nm)
    ax[k].axvline(0, color="k", lw=.6); ax[k].set_xlim(-0.5, 1.5); ax[k].set_title(f"S{SUBJECT} {ch}, average over {len(y)} trials"); ax[k].set_xlabel("time from cue (s)")
ax[0].set_ylabel("microvolts"); ax[0].legend()
# all 40 subjects: F7 minus F8 is the horizontal-gaze signal (opposite sign on the two sides of the head)
i7, i8 = C.idx(names, ["F7"])[0], C.idx(names, ["F8"])[0]
per = {0: [], 1: []}
for s_ in data.SUBJECTS:
    Ds = data.load_subject(s_, causal=True); d = (Ds["X"][:, i7] - Ds["X"][:, i8]).astype(float)
    d = d - d[:, (t >= -0.5) & (t < 0)].mean(1, keepdims=True)
    for lab in (0, 1):
        per[lab].append(d[Ds["y"] == lab].mean(0) * 1e6)
for lab, col, nm in ((0, "tab:blue", "left cue"), (1, "tab:red", "right cue")):
    a = np.array(per[lab]); ax[2].plot(t, a.mean(0), color=col, label=nm); ax[2].fill_between(t, a.mean(0) - a.std(0) / np.sqrt(40), a.mean(0) + a.std(0) / np.sqrt(40), color=col, alpha=.2)
ax[2].axvline(0, color="k", lw=.6); ax[2].set_xlim(-0.5, 1.5); ax[2].set_title("all 40 subjects: F7 minus F8 (mean +- SEM)"); ax[2].set_xlabel("time from cue (s)"); ax[2].legend()
save(fig, "01_f7_f8_by_cue.png")

# %% [markdown]
# ## Splits (nothing from a test trial touches fitting)
# * Within subject: leave-one-run-out. Train on 2 of the 3 imagery runs, test on the third, rotate, pool the predictions.
# * Across subjects: 8 folds of 5 subjects. Train on 35 subjects, test on 5 unseen ones. The scaler is fitted on training subjects only.

# %%
folds = pd.read_csv(OUT / "a3_folds.csv")
print("fold sizes:", folds.fold.value_counts().sort_index().tolist())
print("fold 0 test subjects:", folds[folds.fold == 0].subject.tolist())
fig, ax = plt.subplots(figsize=(9, 2.4))
for r_i, r in enumerate(np.unique(run)):
    for r2_i, r2 in enumerate(np.unique(run)):
        ax.barh(r_i, 1, left=r2_i * 1.05, color="tab:orange" if r == r2 else "tab:gray", edgecolor="w")
ax.set_yticks(range(3)); ax.set_yticklabels([f"fold {i+1}: test run {r}" for i, r in enumerate(np.unique(run))])
ax.set_xticks([0.5, 1.55, 2.6]); ax.set_xticklabels([f"run {r}" for r in np.unique(run)]); ax.invert_yaxis()
ax.set_title("leave-one-run-out (orange = test, grey = train)")
save(fig, "01_loro.png")
