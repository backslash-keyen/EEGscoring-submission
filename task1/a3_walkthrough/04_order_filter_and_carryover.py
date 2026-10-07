# %% [markdown]
# # A3, step 4: three things that surprised me
# 1. The left/right order is NOT random, so the previous trial's label predicts the current one.
# 2. My filter made the "no information" pre-cue window look decodable (a zero-phase filter smears the future into the past).
# 3. Is the frontal result the CURRENT cue or carry-over from the previous trial? A repeat-vs-alternating split decides it.

# %%
from common import *
M = np.full((120, 19), np.nan)          # runs have 6-19 trials (S88/S92/S100 differ), pad with NaN = blank
k = 0
for s in data.SUBJECTS:
    D = data.load_subject(s, causal=True)
    for r in (4, 8, 12):
        yy = D["y"][D["run"] == r]; M[k, :len(yy)] = yy; k += 1
fig, ax = plt.subplots(figsize=(7, 5.2))
ax.imshow(np.ma.masked_invalid(M), aspect="auto", cmap="coolwarm", interpolation="nearest"); ax.set_xlabel("trial in run"); ax.set_ylabel("subject x run (120 rows)")
ax.set_title("blue = left, red = right: stripes mean near-alternation")
save(fig, "04_label_sequences.png")
seq = pd.read_csv(OUT / "a3_sequence.csv")
print(f"consecutive trials with the SAME label: {seq.lag1_same.sum()} observed vs {seq.null_mean.sum():.0f} expected if random")
print("so the previous label alone predicts the current one (pooled accuracy):",
      round(pd.read_csv(OUT / "a3_cross_subject.csv").set_index("decoder").loc["R2", "acc"], 3))

# %% [markdown]
# ## 2) The filter leak, on a synthetic step
# A saccade is a step in the EOG. Zero-phase filtering (filtfilt-like) responds BEFORE the step; minimum-phase filtering does not.

# %%
from mne.filter import filter_data
x = np.zeros(2000); x[1000:] = 1.0
fig, ax = plt.subplots(figsize=(7, 3))
for ph, col in (("zero", "tab:blue"), ("minimum", "tab:orange")):
    y_ = filter_data(x, 160., 1., 40., phase=ph, verbose=False)
    ax.plot((np.arange(2000) - 1000) / 160, y_, color=col, label=f"{ph}-phase")
    print(f"{ph}-phase: first sample above 1% of the step is {(1000 - np.argmax(np.abs(y_) > 0.01)) / 160:.2f} s before the step")
ax.set_xlim(-1.6, 1); ax.axvline(0, color="k", lw=.6); ax.set_xlabel("time (s)"); ax.legend()
save(fig, "04_filter_step.png")

# %%
tc = pd.read_csv(OUT / "a3c_timecourse_with_thresholds.csv")
tw = pd.read_csv(OUT / "a3c_causal_twins_with_thresholds.csv")
print(tw.pivot(index="decoder", columns="filter", values="pooled_xs_acc").round(3).loc[["F1", "F1_pre", "O1", "O1_pre", "N2", "N2_pre"]])
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, nm in zip(axes, ("frontal", "occipital")):
    for flt, col in (("zero-phase", "tab:blue"), ("minimum-phase", "tab:orange")):
        d = tc[(tc.set == nm) & (tc["filter"] == flt)]; ax.plot(d.t, d.acc, color=col, label=flt)
    ax.axvline(0, color="k", lw=.7); ax.axhline(0.5, color="gray", lw=.7); ax.axhline(tc.thr_binom.iloc[0], color="r", ls=":")
    ax.set_title(f"{nm} channels, 100 ms bins, pooled"); ax.set_xlabel("time from cue (s)")
axes[0].set_ylabel("accuracy"); axes[0].legend()
save(fig, "04_timecourse.png")

# %% [markdown]
# ## 3) Current cue or carry-over?
# A carry-over decoder is right on ALTERNATING trials but WRONG on REPEAT trials (it predicts the opposite of last time).
# A decoder that reads the current cue is right on both.

# %%
t = pd.read_csv(OUT / "a3_followup_carryover_with_thresholds.csv")
p = t[(t.regime == "pooled_xs") & (t.scheme == "plain")].set_index("decoder").loc[["F1", "F2", "O1", "N2", "M1", "F1_pre"]]
print(p[["acc_repeat", "thr_repeat", "acc_alternate", "thr_alternate", "acc_balanced", "thr_balanced_normal_approx"]].round(3).to_string())
fig, ax = plt.subplots(figsize=(8, 3.6)); w = .38; xx = np.arange(len(p))
ax.bar(xx - w / 2, p.acc_repeat, w, label=f"repeat trials (n={int(p.rep_n.iloc[0])})")
ax.bar(xx + w / 2, p.acc_alternate, w, label=f"alternating (n={int(p.alt_n.iloc[0])})")
ax.axhline(0.5, color="k", lw=.8); ax.set_xticks(xx); ax.set_xticklabels(p.index); ax.set_ylim(0.4, 0.9); ax.legend(); ax.set_ylabel("pooled accuracy")
ax.set_title("frontal decoders are right on repeats too => current cue, not carry-over")
save(fig, "04_repeat_vs_alternate.png")
