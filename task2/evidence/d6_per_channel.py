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
    allp.append(np.ptp(X, axis=-1) * 1e6)
p = np.concatenate(allp)   # (epochs, channels) in uV
for k,c in enumerate(CHANNELS):
    q=p[:,k]; print(f"{c:10s} median {np.median(q):6.0f}  90% {np.percentile(q,90):6.0f}  99% {np.percentile(q,99):6.0f}  max {q.max():7.0f}  rejected@500: {100*(q>=500).mean():5.1f}%  @1000: {100*(q>=1000).mean():4.1f}%")
eeg=p[:,:2].max(axis=1)
print(f"EEG channels only (Fpz-Cz, Pz-Oz): epochs >=500 uV: {100*(eeg>=500).mean():.2f}%   >=250: {100*(eeg>=250).mean():.2f}%")
