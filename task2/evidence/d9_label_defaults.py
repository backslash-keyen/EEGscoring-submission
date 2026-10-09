"""Defect 9 evidence: y starts as all zeros (= Wake) and STAGE_MAP.get(desc, 0) maps unknown descriptions to Wake."""
import sys, collections, numpy as np, mne
sys.path.insert(0, "..")
from sleep_pipeline import get_files, STAGE_MAP, SUBJECTS, EPOCH_SEC
mne.set_log_level("error")
desc_epochs = collections.Counter(); uncovered = 0; total = 0; per_rec = []
for psg, hyp in get_files(SUBJECTS):
    raw = mne.io.read_raw_edf(psg, stim_channel="Event marker", infer_types=True, preload=False, verbose="error")
    n = raw.n_times // (EPOCH_SEC * int(raw.info["sfreq"]))
    a = mne.read_annotations(hyp)
    covered = np.zeros(n, bool)
    for o, d, desc in zip(a.onset, a.duration, a.description):
        s, e = int(o // EPOCH_SEC), min(int((o + d) // EPOCH_SEC), n)
        covered[s:e] = True; desc_epochs[desc] += max(0, e - s)
        if desc not in STAGE_MAP: per_rec.append((psg.split("\\")[-1][:7], desc, e - s))
    uncovered += int((~covered).sum()); total += n
print("annotation descriptions and epochs they cover:")
for k, v in desc_epochs.most_common(): print(f"  {k:18s} {v:6d}   {'mapped' if k in STAGE_MAP else 'NOT IN STAGE_MAP -> becomes Wake'}")
print(f"epochs with no annotation at all (stay at the default 0 = Wake): {uncovered} of {total}")
print("recordings containing unmapped descriptions:", per_rec)
