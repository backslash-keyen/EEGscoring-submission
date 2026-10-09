"""Defect 10 evidence: how much of the Wake is the long lead-in/lead-out around the sleep period."""
import sys, numpy as np, mne
sys.path.insert(0, "..")
from sleep_pipeline import get_files, load_recording, SUBJECTS
mne.set_log_level("error")
rows = []
for psg, hyp in get_files(SUBJECTS):
    X, y, keep = load_recording(psg, hyp)
    sl = np.where(y > 0)[0]                      # any non-wake stage
    first, last = sl[0], sl[-1]
    rows.append((psg.split("\\")[-1][:7], len(y), int((y == 0).sum()), int((y[:first] == 0).sum()), int((y[last+1:] == 0).sum()),
                 int((y[first:last+1] == 0).sum()), int((y > 0).sum())))
r = np.array([x[1:] for x in rows])
n, w, pre, post, inside, sleep = r.sum(0)
print(f"{len(rows)} recordings, {n} epochs ({n*30/3600:.0f} h)")
print(f"Wake {w} = {100*w/n:.1f}% of all epochs;  sleep stages {sleep} = {100*sleep/n:.1f}%")
print(f"Wake BEFORE first sleep epoch: {pre}  AFTER last sleep epoch: {post}  ({100*(pre+post)/w:.0f}% of all Wake)   Wake inside the sleep period: {inside}")
print(f"always predicting Wake would give accuracy {100*w/(w+sleep):.1f}% (kappa 0)")
print("per recording, lead-in / lead-out Wake in hours (worst 5):")
for x in sorted(rows, key=lambda x: -(x[3]+x[4]))[:5]:
    print(f"  {x[0]}  before {x[3]*30/3600:4.1f} h  after {x[4]*30/3600:4.1f} h  of a {x[1]*30/3600:4.1f} h recording")
k = 60  # 30 min of context kept either side, the usual convention
kept_w = sum(min(x[3], k) + min(x[4], k) + x[5] for x in rows)
print(f"if only 30 min of lead-in/out Wake were kept: Wake {kept_w} epochs, i.e. {100*kept_w/(kept_w+sleep):.1f}% of kept epochs")
