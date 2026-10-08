"""Label-free artefact scan of the epoched data (causal 1-40 Hz cache). Uses no labels, so nothing here can leak into a decoder.
Per subject: bad-channel candidates, trials with large amplitude, blink-like frontal events, temporal high-frequency (EMG) ratio.

python task1/artifact_scan.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).parent))
import data, confound as C

P2P_UV = 100.0   # common rule-of-thumb threshold for an EEG epoch artefact (peak-to-peak, microvolts); not tuned


def scan(s):
    D = data.load_subject(s, causal=True)
    names, X = list(D["ch_names"]), D["X"].astype(np.float64) * 1e6   # microvolts
    n = len(X)
    p2p = np.ptp(X, axis=2)                          # (trials, channels)
    chan_sd = np.median(X.std(2), axis=0)            # typical per-channel amplitude
    robust_z = (chan_sd - np.median(chan_sd)) / (1.4826 * np.median(np.abs(chan_sd - np.median(chan_sd))) + 1e-9)
    bad = [names[i] for i in np.where((robust_z > 4) | (chan_sd < 0.5))[0]]
    frontal = C.idx(names, ["Fp1", "Fpz", "Fp2"])
    heog = X[:, C.idx(names, ["F7"])[0]] - X[:, C.idx(names, ["F8"])[0]]   # F7-F8 as a horizontal-eye-movement proxy
    f, Pw = welch(X[:, C.idx(names, C.TEMP)], fs=data.FS, nperseg=128, axis=-1)
    hf = Pw[:, :, (f >= 30) & (f <= 40)].mean((1, 2)) / Pw[:, :, (f >= 8) & (f <= 13)].mean((1, 2))
    return dict(subject=s, n_trials=n,
                median_p2p_uv=float(np.median(p2p)),
                frac_trials_any_ch_p2p_gt100=float((p2p.max(1) > P2P_UV).mean()),
                frac_trials_frontal_p2p_gt100=float((p2p[:, frontal].max(1) > P2P_UV).mean()),
                frac_trials_nonfrontal_p2p_gt100=float((np.delete(p2p, frontal, axis=1).max(1) > P2P_UV).mean()),
                heog_p2p_median_uv=float(np.median(np.ptp(heog, axis=1))),
                n_bad_channels=len(bad), bad_channels=" ".join(bad),
                temporal_30_40_over_mu_median=float(np.median(hf)))


def main():
    df = pd.DataFrame([scan(s) for s in data.SUBJECTS])
    df.to_csv(data.ROOT / "outputs" / "a3d_artifact_scan.csv", index=False)
    pd.set_option("display.width", 220)
    print(df.drop(columns="bad_channels").describe().loc[["min", "50%", "max"]].round(3).T)
    print("\nsubjects with >20% trials containing any channel p2p > 100 uV:", df[df.frac_trials_any_ch_p2p_gt100 > .2].subject.tolist())
    print("subjects with bad-channel candidates:", df[df.n_bad_channels > 0][["subject", "bad_channels"]].values.tolist())
    print("channels flagged, counted over subjects:", pd.Series(" ".join(df.bad_channels).split()).value_counts().to_dict())


if __name__ == "__main__":
    main()
