"""A2 (part 1) - Time-frequency power around the cue at C3, C4 and neighbours, relative to a justified baseline.

Brief: "Per subject, compute time-frequency power around the imagery cue at C3, C4 and neighbours, relative to a
baseline you justify."

Live part: Morlet power (task1/erd.py, same wavelets as the pipeline) for SUBJECT at C3, C4 and their 4 nearest
neighbours, each on the Laplacian reference (A2b explains why), shown as ERD% = power / baseline - 1.
Change SUBJECT and re-run. All 40 subjects' C3/C4 maps are in outputs/erd/S0xx.png (written by task1/erd.py).
Run: python task1/asks/A2a_time_frequency_power.py
"""
from _common import *

SUBJECT = 72
REF = "laplacian"          # try "raw" or "car" to see what the reference does to the maps (A2b)

header("A2.1  Time-frequency power at C3, C4 and neighbours",
       "per-subject time-frequency power around the imagery cue at C3, C4 and neighbours, relative to a justified baseline.",
       "Morlet wavelets 6-30 Hz (n_cycles = f/2) on every trial; baseline = -1.0..-0.1 s before the cue, pooled over trials "
       "of both classes (label-blind); ERD% = power / baseline - 1, averaged per imagined hand.")

print_decision("D5")
print_decision("D10")

if has_data():
    data, erd = pipeline("data", "erd")
    D = data.load_subject(SUBJECT)
    X, y, names = D["X"].astype(np.float64), D["y"], list(D["ch_names"])
    times = np.arange(data.N_T) / data.FS + data.TMIN
    base_m = (times >= data.T_BASE[0]) & (times <= data.T_BASE[1])
    groups = {c: [c] + [str(names[i]) for i in erd.neighbours(names, c)] for c in ("C3", "C4")}
    section(f"S{SUBJECT:03d}: {int((y == 0).sum())} left / {int((y == 1).sum())} right trials; sites = {groups}")

    fig, axes = plt.subplots(4, 5, figsize=(14, 9), sharex=True, sharey=True)
    rows, per_trial = [], {}
    for gi, (centre, sites) in enumerate(groups.items()):
        for si, ch in enumerate(sites):
            P = erd.power(erd.derive(X, names, ch, REF))                 # (trials, freqs, times)
            base = P[:, :, base_m].mean((0, 2))                          # one baseline per frequency, both classes
            for ci, (cls, hand) in enumerate(((0, "left"), (1, "right"))):
                m = P[y == cls].mean(0) / base[:, None] - 1
                ax = axes[2 * gi + ci, si]
                im = ax.imshow(m * 100, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-60, vmax=60,
                               extent=[times[0], times[-1], erd.FREQS[0], erd.FREQS[-1]])
                ax.axvline(0, color="k", lw=.6)
                ax.set_title(f"{ch} | {hand} imagery", fontsize=8)
            for band, (lo, hi) in erd.BANDS.items():
                e = erd.band_erd_db(P, times, lo, hi)                    # per trial, baseline pooled over all trials
                per_trial[(ch, band)] = e
                rows += [dict(site=ch, imagery=hand, band=band, erd_db=float(e[y == cls].mean()))
                         for cls, hand in ((0, "left"), (1, "right"))]
    fig.colorbar(im, ax=axes, shrink=.6, label="ERD %  (blue = power drop vs baseline)")
    fig.supxlabel("time from cue (s)"); fig.supylabel("frequency (Hz)")
    fig.suptitle(f"A2.1  S{SUBJECT:03d}, {REF} reference: rows 1-2 = C3 and neighbours, rows 3-4 = C4 and neighbours", fontsize=10)
    save(fig, f"A2a_tfr_S{SUBJECT:03d}_{REF}.png")

    section("Band ERD in dB, 0.5-4.0 s vs baseline (negative = desynchronisation)")
    t = pd.DataFrame(rows).pivot_table(index="site", columns=["band", "imagery"], values="erd_db")
    print(t.reindex(groups["C3"] + groups["C4"]).round(2).to_string())
    say("Expected for lateralised motor imagery: C4 drops more for LEFT-hand imagery and C3 more for RIGHT-hand imagery.")

    if REF == "laplacian":   # the pipeline's own numbers for this subject, to show the live result is the saved one
        saved = csv("a2_lateralisation.csv").set_index("subject").loc[SUBJECT]
        for band in erd.BANDS:
            contra = np.where(y == 0, per_trial[("C4", band)], per_trial[("C3", band)]).mean()   # C4 for left, C3 for right
            check(f"{band} contralateral ERD (dB)", contra, saved[f"{band}_contra_db"], tol=1e-6)
else:
    no_data_note()
    show_png(f"erd/S{SUBJECT:03d}.png", f"A2.1  S{SUBJECT:03d} C3/C4 ERD maps (saved by task1/erd.py)")

finish()
