"""Parallel pre-download of the Sleep-EDF files that fetch_data(subjects 0-14, recording 1,2) needs.

Why: physionet.org throttles each connection to ~30 kB/s, so MNE's sequential download of ~1.4 GB takes
many hours. MNE skips a file that is already on disk with the right SHA1, so filling the cache
in parallel does not change what sleep_pipeline.py loads."""
import hashlib, os, sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from mne.datasets.sleep_physionet import age as A

DEST = os.environ["MNE_DATA"] + "/physionet-sleep-data"
BASE = "https://physionet.org/physiobank/database/sleep-edfx/sleep-cassette/"
os.makedirs(DEST, exist_ok=True)

rec = np.loadtxt(os.path.join(os.path.dirname(A.__file__), "age_records.csv"), skiprows=1, delimiter=",",
                 usecols=(0, 1, 2, 6, 7),
                 dtype={"names": ("subject", "record", "type", "sha", "fname"),
                        "formats": ("<i2", "i1", "<S9", "S40", "<S22")})
want = [(r["fname"].decode(), r["sha"].decode()) for r in rec if r["subject"] < 15]


def sha1(p):
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def get(item):
    fname, sha = item
    out = os.path.join(DEST, fname)
    if os.path.exists(out) and sha1(out) == sha:
        return fname, "cached"
    tmp = out + ".prefetch"
    for attempt in range(40):
        try:
            have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            # resume with a Range request: restarting from zero on every dropped connection never finishes
            hdr = {"Range": f"bytes={have}-"} if have else {}
            with requests.get(BASE + fname, headers=hdr, stream=True, timeout=(30, 60)) as r:
                if r.status_code == 416:  # already complete
                    pass
                else:
                    r.raise_for_status()
                    with open(tmp, "ab" if r.status_code == 206 else "wb") as f:
                        for chunk in r.iter_content(1 << 16):
                            f.write(chunk)
            if sha1(tmp) == sha:
                os.replace(tmp, out)
                return fname, "ok"
            os.remove(tmp)  # corrupt or wrong-size partial: start over
        except Exception as e:  # throttling / dropped connection: keep the partial and resume
            print("retry", fname, attempt, type(e).__name__, flush=True)
    return fname, "FAILED"


if __name__ == "__main__":
    print(len(want), "files", flush=True)
    with ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 16) as ex:
        for fname, st in ex.map(get, want):
            print(fname, st, flush=True)
