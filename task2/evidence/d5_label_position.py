"""Defect 5 evidence: the head reads the CENTRE epoch (position SEQ_LEN // 2) but the given script labels the
window with its LAST epoch, y[i + SEQ_LEN - 1]. Counts how often the two labels differ, from the hypnograms."""
import collections, sys, numpy as np, mne
sys.path.insert(0, "..")
from sleep_pipeline import get_files, SUBJECTS, STAGE_MAP, EPOCH_SEC, SEQ_LEN, CLASS_NAMES, WAKE_MARGIN
mne.set_log_level("error")
c = SEQ_LEN // 2
for crop in (False, True):
    tot = diff = 0; by = collections.Counter(); by_diff = collections.Counter()
    for psg, hyp in get_files(SUBJECTS):
        raw = mne.io.read_raw_edf(psg, stim_channel="Event marker", infer_types=True, preload=False, verbose="error")
        n = raw.n_times // (EPOCH_SEC * int(raw.info["sfreq"]))
        y = np.full(n, -1)
        a = mne.read_annotations(hyp)
        for o, d, desc in zip(a.onset, a.duration, a.description):
            y[int(o // EPOCH_SEC):int((o + d) // EPOCH_SEC)] = STAGE_MAP.get(desc, -1)
        if crop:                                                   # the fixed pipeline's sleep-period crop
            s = np.where(y > 0)[0]; y = y[max(0, s[0] - WAKE_MARGIN): s[-1] + 1 + WAKE_MARGIN]
        for i in range(len(y) - SEQ_LEN + 1):
            yc, yl = y[i + c], y[i + SEQ_LEN - 1]
            if yc < 0:
                continue
            tot += 1; by[yc] += 1
            if yl != yc:
                diff += 1; by_diff[yc] += 1
    print(f"{'cropped to sleep +- 30 min' if crop else 'whole recordings (as given)'}: windows {tot}, "
          f"last-epoch label != centre label in {diff} ({100 * diff / tot:.1f}%)")
    print("   by centre stage: " + ", ".join(f"{CLASS_NAMES[k]} {100 * by_diff[k] / by[k]:.1f}%" for k in range(5)))
