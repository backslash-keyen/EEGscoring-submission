"""A1 data audit: one row per subject -> audit.csv, plus the alpha-peak test for the 128 Hz header."""
import re, glob
from pathlib import Path
import csv
import numpy as np, mne
from scipy.signal import welch

mne.set_log_level("ERROR")
ROOT = Path(__file__).resolve().parents[1]
RUNS = [4, 8, 12]
# Event meaning for imagery runs 4/8/12 (see wiki/dataset.md for where verified).
CODE_MEANING = "T0=rest,T1=imagine left fist,T2=imagine right fist"
NOMINAL_FS, EXPECTED_PER_CLASS, CUE_NOMINAL = 160.0, 7.5, 4.1
# Posterior sites carry the occipital/parietal alpha rhythm; used only to estimate the true rate.
ALPHA_CHS = ["O1..", "O2..", "Oz..", "P3..", "P4..", "Pz.."]


# Decisions per anomalous subject; reasons live in DECISIONS.md. Nothing is dropped silently.
_RES = ("KEEP; resample 128->160 Hz in preprocessing. Alpha-peak check (header rate) does not show the 0.8x shift a "
        "mislabelled 160 Hz file would give for all three, and 128 Hz runs have 19 trials/run at 5.12 s cues, i.e. a different "
        "protocol rather than a wrong header")
ACTIONS = {
    88: _RES, 92: _RES,
    100: _RES + "; only 6 L + 6 R per run (18 each), kept, flagged for low per-subject trial count",
    104: "KEEP; drop only the final T1 trial of run 8 (cue 2.98 s: recording ends mid-cue, shorter than the analysis window)",
}


def alpha_peak(raw):
    """Peak frequency (6-14 Hz) of the rest (T0) spectrum using the header rate."""
    fs = raw.info["sfreq"]
    ann = raw.annotations
    picks = [raw.ch_names.index(c) for c in ALPHA_CHS]
    X = raw.get_data(picks=picks)
    segs = [X[:, int(o * fs):int((o + d) * fs)] for o, d, e in zip(ann.onset, ann.duration, ann.description) if e == "T0"]
    psd = []
    for s in segs:
        # nperseg of 2 s keeps ~0.5 Hz resolution, enough to separate 128/160 = 0.8x shift (~2 Hz at 10 Hz)
        f, p = welch(s, fs=fs, nperseg=int(2 * fs))
        psd.append(p.mean(0))
    f, p = f, np.mean(psd, 0)
    m = (f >= 6) & (f <= 14)
    return float(f[m][np.argmax(p[m])])


def audit_subject(s):
    rows, fs_set, per_run, anomalies = [], set(), [], []
    t1 = t2 = 0
    durs, cues, peaks = [], [], []
    for r in RUNS:
        f = ROOT / f"data/MNE-eegbci-data/files/eegmmidb/1.0.0/S{s:03d}/S{s:03d}R{r:02d}.edf"
        raw = mne.io.read_raw_edf(f, preload=True)
        fs = raw.info["sfreq"]; fs_set.add(fs)
        a = raw.annotations
        d = list(a.description)
        n1, n2 = d.count("T1"), d.count("T2")
        t1 += n1; t2 += n2; durs.append(float(raw.times[-1]))
        cues += [x for x, e in zip(a.duration, d) if e in ("T1", "T2")]
        peaks.append(alpha_peak(raw))
        if set(d) - {"T0", "T1", "T2"}: anomalies.append(f"R{r}: unexpected codes {set(d)-{'T0','T1','T2'}}")
        if n1 + n2 < 14 or n1 + n2 > 16 and fs == NOMINAL_FS: anomalies.append(f"R{r}: {n1}L/{n2}R trials")
        sd = raw.get_data().std(1) * 1e6
        if (sd < 0.1).any(): anomalies.append(f"R{r}: flat channel(s)")
    fs = sorted(fs_set)
    if fs != [NOMINAL_FS]: anomalies.append(f"header rate {fs} Hz")
    if max(durs) - min(durs) > 5: anomalies.append(f"run durations {[round(x,1) for x in durs]} s differ")
    odd = sorted({round(float(c), 2) for c in cues if abs(c - CUE_NOMINAL) > 0.2})
    if odd and fs == [NOMINAL_FS]: anomalies.append(f"cue lengths {odd} s")
    return dict(subject=s, runs_loaded="4,8,12", sfreq_hz=fs[0] if len(fs) == 1 else str(fs),
                duration_s=round(float(sum(durs)), 1), n_left=t1, n_right=t2,
                event_code_meaning=CODE_MEANING, cue_s=sorted({round(float(c), 2) for c in cues}).__repr__(),
                alpha_peak_hz=round(float(np.mean(peaks)), 2), anomalies="; ".join(anomalies) or "none",
                action=ACTIONS.get(s, "kept as is"))


def add_usage(rows):
    """Trials actually analysed vs trials annotated: windows that run past the end of the recording are dropped (never silently)."""
    import sys; sys.path.insert(0, str(ROOT / "task1"))
    import data
    for r in rows:
        d = data.load_subject(r["subject"])
        r["n_left_used"], r["n_right_used"] = int((d["y"] == 0).sum()), int((d["y"] == 1).sum())
        dropped = r["n_left"] + r["n_right"] - r["n_left_used"] - r["n_right_used"]
        r["n_dropped"] = dropped
        if dropped:
            r["action"] += f"; {dropped} trial(s) dropped: analysis window (-1.5..4.0 s) runs past recording end (run ends mid-cue)"


if __name__ == "__main__":
    rows = [audit_subject(s) for s in range(70, 110)]
    add_usage(rows)
    with open(ROOT / "audit.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print({k: v for k, v in r.items() if k != "event_code_meaning"})
    odd = [r for r in rows if r["sfreq_hz"] != 160.0]
    rest = [r["alpha_peak_hz"] for r in rows if r["sfreq_hz"] == 160.0]
    print("\nalpha peak (header rate): 128 Hz subjects", [(r["subject"], r["alpha_peak_hz"]) for r in odd],
          "| 160 Hz subjects mean %.2f sd %.2f" % (np.mean(rest), np.std(rest)))
    print("totals L/R:", sum(r["n_left"] for r in rows), sum(r["n_right"] for r in rows))
