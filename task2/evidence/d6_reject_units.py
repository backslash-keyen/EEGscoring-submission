"""Defect 6 evidence: REJECT_PTP = 500 (uV) is compared against raw.get_data(), which MNE returns in VOLTS."""
import sys, numpy as np, mne
sys.path.insert(0, "..")
from sleep_pipeline import get_files, SUBJECTS, CHANNELS, EPOCH_SEC, REJECT_PTP
mne.set_log_level("error")
allp = []; n_ep = 0
for psg, hyp in get_files(SUBJECTS):
    raw = mne.io.read_raw_edf(psg, stim_channel="Event marker", infer_types=True, preload=True, verbose="error")
    raw.pick(CHANNELS); raw.filter(0.3, 35.0, verbose="error")
    sf = int(raw.info["sfreq"]); n = raw.n_times // (EPOCH_SEC * sf)
    X = raw.get_data()[:, : n * EPOCH_SEC * sf].reshape(len(CHANNELS), n, -1).transpose(1, 0, 2)
    allp.append(np.ptp(X, axis=-1).max(axis=1))          # worst channel per epoch, in volts
p = np.concatenate(allp)
print(f"epochs: {len(p)}")
print(f"peak-to-peak of the worst channel, in volts : median {np.median(p):.2e}  99.9% {np.percentile(p,99.9):.2e}  max {p.max():.2e}")
print(f"threshold in the code                       : REJECT_PTP = {REJECT_PTP} compared against these numbers as-is")
print(f"epochs rejected by the code as written      : {(p >= REJECT_PTP).sum()}  (the largest value is {p.max():.2e}, 500 is never reached)")
pu = p * 1e6
print(f"in uV: median {np.median(pu):.0f}  99.9% {np.percentile(pu,99.9):.0f}  max {pu.max():.0f}")
for t in (500, 1000):
    print(f"epochs a {t} uV threshold WOULD reject      : {(pu >= t).sum()}  ({100*(pu >= t).mean():.2f}%)")
