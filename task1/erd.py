"""A2: per-subject ERD at C3/C4, lateralisation index (mu, beta), single-subject permutation test, present/absent label."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mne.time_frequency import tfr_array_morlet

sys.path.insert(0, str(Path(__file__).parent))
import data

OUT = data.ROOT / "outputs"
(OUT / "erd").mkdir(parents=True, exist_ok=True)
FREQS = np.arange(6.0, 31.0, 1.0)
BANDS = {"mu": (8, 13), "beta": (13, 30)}
ACTIVE = (0.5, 4.0)    # skip the first 0.5 s: cue-evoked visual response and ERD onset lag (DECISIONS D5)
N_PERM = 10000
ALPHA = 0.05 / 2       # Bonferroni across the two bands: "present" if either band passes
REFS = ["raw", "car", "laplacian"]


def neighbours(ch_names, ch, k=4):
    pos = data.montage_pos(ch_names)
    i = list(ch_names).index(ch)
    d = np.linalg.norm(pos - pos[i], axis=1)
    d[i] = np.inf
    return list(np.argsort(d)[:k])


def derive(X, ch_names, ch, ref):
    """Signal at `ch` under a spatial reference. Laplacian = ch minus mean of 4 nearest electrodes: any term common to
    all of them (including the unknown recording reference) cancels because the weights sum to zero (DECISIONS D4)."""
    i = list(ch_names).index(ch)
    if ref == "raw":
        return X[:, i]
    if ref == "car":
        return X[:, i] - X.mean(1)
    return X[:, i] - X[:, neighbours(ch_names, ch)].mean(1)


def power(sig):
    # n_cycles = f/2 keeps ~constant relative bandwidth; adequate resolution for 1 Hz steps in mu/beta
    return tfr_array_morlet(sig[:, None, :], data.FS, FREQS, n_cycles=FREQS / 2, output="power", n_jobs=1)[:, 0]


def band_erd_db(P, times, lo, hi):
    """Per-trial ERD in dB: active-window band power vs subject-pooled pre-cue baseline (label-blind, so no leakage)."""
    f = (FREQS >= lo) & (FREQS <= hi)
    base = P[:, f][:, :, (times >= data.T_BASE[0]) & (times <= data.T_BASE[1])].mean()
    act = P[:, f][:, :, (times >= ACTIVE[0]) & (times <= ACTIVE[1])].mean((1, 2))
    return 10 * np.log10(act / base)


def perm_p(d, y, rng):
    """One-sided label-permutation p for T = mean(d|L) - mean(d|R) where d = C3 - C4 ERD (expected > 0 for contralateral ERD)."""
    stat = d[y == 0].mean() - d[y == 1].mean()
    null = np.empty(N_PERM)
    for k in range(N_PERM):
        yp = rng.permutation(y)
        null[k] = d[yp == 0].mean() - d[yp == 1].mean()
    return stat, (1 + (null >= stat).sum()) / (N_PERM + 1)


def analyse(s, ref, keep_tfr=False):
    D = data.load_subject(s)
    X, y, names = D["X"].astype(np.float64), D["y"], D["ch_names"]
    times = np.arange(data.N_T) / data.FS + data.TMIN
    P = {ch: power(derive(X, names, ch, ref)) for ch in ("C3", "C4")}
    row = dict(subject=s, ref=ref, n_left=int((y == 0).sum()), n_right=int((y == 1).sum()))
    rng = np.random.default_rng(s)   # per-subject seed => reproducible p-values
    for band, (lo, hi) in BANDS.items():
        e3, e4 = band_erd_db(P["C3"], times, lo, hi), band_erd_db(P["C4"], times, lo, hi)
        contra = np.where(y == 0, e4, e3)   # left-hand imagery -> right hemisphere (C4)
        ipsi = np.where(y == 0, e3, e4)
        stat, p = perm_p(e3 - e4, y, rng)
        erd_c, erd_i = contra.mean(), ipsi.mean()
        row.update({f"{band}_contra_db": erd_c, f"{band}_ipsi_db": erd_i,
                    f"{band}_LI": (erd_i - erd_c) / (abs(erd_i) + abs(erd_c)),  # >0: contralateral desynchronises more
                    f"{band}_T": stat, f"{band}_p": p})
    row["label"] = "present" if any((row[f"{b}_p"] < ALPHA) and (row[f"{b}_contra_db"] < 0) for b in BANDS) else "absent"
    return (row, P, y, times) if keep_tfr else row


def plot_subject(s, row, P, y, times, ax_row=None, title=True):
    """ERD% maps, rows = cue class, cols = C3/C4; pooled baseline."""
    fig = None
    if ax_row is None:
        fig, ax_row = plt.subplots(2, 2, figsize=(8, 5), sharex=True, sharey=True)
    for ci, ch in enumerate(("C3", "C4")):
        base = P[ch][:, :, (times >= data.T_BASE[0]) & (times <= data.T_BASE[1])].mean((0, 2))
        for ri, (cls, nm) in enumerate(((0, "left"), (1, "right"))):
            m = P[ch][y == cls].mean(0) / base[:, None] - 1
            ax = ax_row[ri, ci]
            ax.imshow(m * 100, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-100, vmax=100,
                      extent=[times[0], times[-1], FREQS[0], FREQS[-1]])
            ax.axvline(0, color="k", lw=.6)
            ax.set_title(f"{nm} imagery | {ch}", fontsize=8)
    if fig is not None:
        fig.suptitle(f"S{s} ({row['label']}) mu p={row['mu_p']:.3f} beta p={row['beta_p']:.3f} (Laplacian ref)", fontsize=9)
        fig.supxlabel("time from cue (s)"); fig.supylabel("frequency (Hz)")
        fig.savefig(OUT / "erd" / f"S{s:03d}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)


def main():
    rows, tfr = [], {}
    for s in data.SUBJECTS:
        for ref in REFS:
            keep = ref == "laplacian"
            res = analyse(s, ref, keep_tfr=keep)
            if keep:
                row, P, y, times = res
                tfr[s] = (P, y, times)
                plot_subject(s, row, P, y, times)
            else:
                row = res
            rows.append(row)
        print(s, [r["label"] for r in rows[-3:]], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "a2_all_references.csv", index=False)
    prim = df[df.ref == "laplacian"].drop(columns="ref")
    prim.to_csv(OUT / "a2_lateralisation.csv", index=False)

    # 3 present + 3 absent by a stated rule, not hand-picked. Present: most negative contralateral ERD among significant bands
    # (ranking by LI alone picks subjects whose ipsilateral synchronisation is an artefact, e.g. S95); absent: highest p.
    def clean_erd(r):
        return min(r[f"{b}_contra_db"] for b in BANDS if r[f"{b}_p"] < ALPHA)
    pp = prim[prim.label == "present"]
    pres = sorted(pp.subject, key=lambda s_: clean_erd(pp[pp.subject == s_].iloc[0]))[:3]
    absn = prim[prim.label == "absent"].assign(pm=lambda d: d[["mu_p", "beta_p"]].min(axis=1)).sort_values("pm", ascending=False).subject.head(3).tolist()
    fig, axes = plt.subplots(6, 4, figsize=(11, 14), sharex=True, sharey=True)
    for k, s in enumerate(pres + absn):
        P, y, times = tfr[s]
        row = prim[prim.subject == s].iloc[0].to_dict()
        # one subject per row of 4 panels: (left|C3, left|C4, right|C3, right|C4)
        for ci, ch in enumerate(("C3", "C4")):
            base = P[ch][:, :, (times >= data.T_BASE[0]) & (times <= data.T_BASE[1])].mean((0, 2))
            for ri, cls in enumerate((0, 1)):
                m = P[ch][y == cls].mean(0) / base[:, None] - 1
                ax = axes[k, 2 * ri + ci]
                ax.imshow(m * 100, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-100, vmax=100,
                          extent=[times[0], times[-1], FREQS[0], FREQS[-1]])
                ax.axvline(0, color="k", lw=.6)
                ax.set_title(f"S{s} {row['label']} | {'LR'[ri]} imagery | {ch}", fontsize=7)
    fig.supxlabel("time from cue (s)"); fig.supylabel("frequency (Hz)")
    fig.savefig(OUT / "a2_examples_3present_3absent.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    print("\nlabels (Laplacian):", prim.label.value_counts().to_dict())
    print("selected examples present:", pres, "absent:", absn)
    for ref in REFS:
        d = df[df.ref == ref]
        print(ref, "present:", int((d.label == "present").sum()), "| mean mu LI %.3f beta LI %.3f" % (d.mu_LI.mean(), d.beta_LI.mean()))


if __name__ == "__main__":
    main()
