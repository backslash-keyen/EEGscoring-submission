"""Defect 8 evidence: absolute amplitude differs by stage (N3 is DEFINED by >75 uV slow waves), and per-epoch z-scoring erases it."""
import sys, numpy as np, mne
sys.path.insert(0, "..")
from sleep_pipeline import get_files, SUBJECTS, CHANNELS, EPOCH_SEC, STAGE_MAP, CLASS_NAMES, load_recording
mne.set_log_level("error")
rows = {k: [] for k in range(5)}; z_std = []
for psg, hyp in get_files(SUBJECTS):
    X, y, keep = load_recording(psg, hyp)                      # already z-scored per epoch/channel (as the pipeline does)
    z_std.append(X.std(axis=-1).mean())
    raw = mne.io.read_raw_edf(psg, stim_channel="Event marker", infer_types=True, preload=True, verbose="error")
    raw.pick(["Fpz-Cz"]); raw.filter(0.3, 35.0); raw_d = raw.copy().filter(0.5, 2.0)       # slow-wave band
    sf = int(raw.info["sfreq"]); n = raw.n_times // (EPOCH_SEC * sf)
    # recompute the same crop as load_recording to align labels with epochs
    a = mne.read_annotations(hyp); yy = np.full(n, -1)
    for o, d, desc in zip(a.onset, a.duration, a.description): yy[int(o//EPOCH_SEC):int((o+d)//EPOCH_SEC)] = STAGE_MAP.get(desc, -1)
    asleep = np.where(yy > 0)[0]; lo, hi = max(0, asleep[0]-60), min(n, asleep[-1]+61)
    D = raw_d.get_data()[0, : n*EPOCH_SEC*sf].reshape(n, -1)[lo:hi] * 1e6
    for k in range(5):
        rows[k].append(np.ptp(D[yy[lo:hi] == k], axis=-1))
print("Fpz-Cz, 0.5-2 Hz slow-wave band, peak-to-peak per epoch (uV), raw amplitude BEFORE z-scoring:")
print(f"{'stage':5s} {'epochs':>7s} {'median':>7s} {'>75 uV':>8s}")
for k, n_ in enumerate(CLASS_NAMES):
    v = np.concatenate(rows[k]); print(f"{n_:5s} {len(v):7d} {np.median(v):7.0f} {100*np.mean(v > 75):7.1f}%")
print(f"after the pipeline's z-scoring every epoch/channel has std = {np.mean(z_std):.3f}: absolute amplitude is identical for all stages")
