"""Shared loader: all imagery trials of subjects 70-109 as one cached array set (raw reference, 1-40 Hz, 160 Hz)."""
from pathlib import Path
import numpy as np, mne

mne.set_log_level("ERROR")
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
CACHE_CAUSAL = ROOT / "data" / "cache_causal"   # minimum-phase filter: nothing spreads backwards in time (DECISIONS D14)
RUNS = [4, 8, 12]            # imagery left/right fist; verified in wiki/dataset.md
SUBJECTS = list(range(70, 110))
FS = 160.0
TMIN, TMAX = -1.5, 4.0       # shortest regular cue is 4.0 s; S104's truncated 2.98 s cue is dropped (DECISIONS D3)
N_T = int((TMAX - TMIN) * FS) + 1   # fixed sample count so every epoch has identical shape
T_BASE = (-1.0, -0.1)        # baseline window inside the preceding rest (DECISIONS D5)


def edf(s, r):
    return ROOT / f"data/MNE-eegbci-data/files/eegmmidb/1.0.0/S{s:03d}/S{s:03d}R{r:02d}.edf"


def load_subject(s, causal=False):
    """Return dict X (n,64,T) float32 in volts, y (0=left,1=right), run, trial (index within run), onset (s), ch_names."""
    cache = CACHE_CAUSAL if causal else CACHE
    f = cache / f"S{s:03d}.npz"
    if f.exists():
        d = np.load(f, allow_pickle=True)
        return {k: d[k] for k in d.files}
    Xs, ys, runs, trials, onsets = [], [], [], [], []
    for r in RUNS:
        raw = mne.io.read_raw_edf(edf(s, r), preload=True)
        raw.rename_channels(lambda n: n.strip("."))
        # 1-40 Hz: removes drift and line noise before wavelets/decoding; applied to continuous data so epochs have no edge artefacts
        raw.filter(1.0, 40.0, phase="minimum" if causal else "zero", verbose=False)
        if raw.info["sfreq"] != FS:
            raw.resample(FS)   # 128 Hz-header subjects (88, 92, 100): DECISIONS D2
        ev, _ = mne.events_from_annotations(raw, event_id={"T1": 1, "T2": 2})
        ann = raw.annotations
        dur = {round(o, 3): d for o, d, e in zip(ann.onset, ann.duration, ann.description) if e in ("T1", "T2")}
        k = 0
        for sample, _, code in ev:
            t = sample / FS
            a = int(round(sample + TMIN * FS))
            if a < 0 or a + N_T > raw.n_times:   # window must fit in the recording (drops S104 R8 last trial)
                continue
            seg = raw.get_data(start=a, stop=a + N_T)
            Xs.append(seg.astype(np.float32)); ys.append(code - 1); runs.append(r); trials.append(k); onsets.append(t); k += 1
    out = dict(X=np.stack(Xs), y=np.array(ys), run=np.array(runs), trial=np.array(trials),
               onset=np.array(onsets), ch_names=np.array(raw.ch_names))
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **out)
    return out


def montage_pos(ch_names):
    """Unit-sphere-ish 3D positions (m) of the channels from standard_1005, for references and later interpolation."""
    m = mne.channels.make_standard_montage("standard_1005")
    p = m.get_positions()["ch_pos"]
    lower = {k.lower(): v for k, v in p.items()}
    return np.array([lower[c.lower()] for c in ch_names])


def load_all(causal=False):
    return {s: load_subject(s, causal) for s in SUBJECTS}
