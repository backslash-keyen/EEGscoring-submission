"""Task 2d: which stage the fixed model gets worst, and why, from the signals Sleep-EDF SC actually records.

  python task2/physiology.py --step facts      channels / rates / durations / label vocabulary from the files themselves
  python task2/physiology.py --step features   per-epoch physiology in uV for every (cropped) epoch of all 29 recordings
  python task2/physiology.py --step predict    train the fixed pipeline per seed and keep its per-window test predictions
  python task2/physiology.py --step report     per-stage metrics, confusions, feature profiles, figures
  python task2/physiology.py                   all four, skipping any step whose output already exists

Outputs in task2/physiology/. The model is the unmodified task2/sleep_pipeline.py: `predict` and `load_recording` are
wrapped at run time only to record which file each test window came from and what the model answered.
"""
import argparse, collections, os, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "physiology"
SEEDS = [42, 43, 44]          # same seeds as the impact ledger, so the fixed-pipeline runs can be cross-checked
STAGES = ["W", "N1", "N2", "N3", "REM"]
os.environ.setdefault("MNE_DATA", str(HERE.parent / "data" / "mne_sleep"))
sys.path.insert(0, str(HERE))

import mne                                   # noqa: E402
import sleep_pipeline as sp                  # noqa: E402

mne.set_log_level("error")


def files():
    return sp.get_files(sp.SUBJECTS)


def name_of(psg):
    return Path(psg).name[:7]                 # SC4ssnE: subject ss, night n


# ---------------------------------------------------------------------------------------------------------------
# facts: what the files contain (the dataset page is quoted in task2/PHYSIOLOGY.md)
# ---------------------------------------------------------------------------------------------------------------
def step_facts():
    sig, desc, hours = collections.Counter(), collections.Counter(), []
    for psg, hyp in files():
        raw = mne.io.read_raw_edf(psg, preload=False)
        ex = raw._raw_extras[0]
        sig[tuple(f"{c} @ {ex['n_samps'][i] / ex['record_length'][0]:g} Hz" for i, c in enumerate(raw.ch_names))] += 1
        hours.append(raw.times[-1] / 3600)
        a = mne.read_annotations(hyp)
        for d, du in zip(a.description, a.duration):
            desc[d] += int(du // sp.EPOCH_SEC)
    lines = [f"{len(hours)} PSG files (subjects 0-14, both nights; subject 13 night 2 absent from the corpus)", ""]
    lines += [f"{n} files with signals: " + ", ".join(k) for k, n in sig.items()]
    lines += [f"recording length {min(hours):.1f}-{max(hours):.1f} h", "",
              "hypnogram labels (30-s epochs): " + ", ".join(f"{d}: {n}" for d, n in sorted(desc.items())),
              f"pipeline uses: {sp.CHANNELS} (EMG submental, resp, temp are in the file but not used)"]
    (OUT / "dataset_facts.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---------------------------------------------------------------------------------------------------------------
# features: R&K criteria measured in uV on the same 30-s epochs and the same crop the pipeline uses
# ---------------------------------------------------------------------------------------------------------------
def band_power(f, p, lo, hi):
    m = (f >= lo) & (f < hi)
    return p[..., m].sum(-1) * (f[1] - f[0])


def delta_cover(x, fs):
    # R&K stage 3/4: share of the epoch with 0.5-2 Hz waves of >75 uV peak-to-peak (3: 20-50%, 4: >50%).
    # Approximation: peak-to-peak of the 0.5-2 Hz band in 1-s steps, each step 'covered' if > 75 uV.
    from scipy.signal import butter, sosfiltfilt
    d = sosfiltfilt(butter(4, [0.5, 2.0], "bandpass", fs=fs, output="sos"), x, axis=-1)
    seg = d.reshape(d.shape[0], -1, fs)       # (epochs, 30, 1 s)
    return (np.ptp(seg, axis=-1) > 75).mean(-1)


def step_features():
    from scipy.signal import welch
    rows = []
    for psg, hyp in files():
        # same read and filter as sp.load_recording, but kept in uV (the pipeline z-scores each epoch afterwards)
        raw = mne.io.read_raw_edf(psg, stim_channel="Event marker", infer_types=True, preload=True)
        fs = int(raw.info["sfreq"])
        n_ep = raw.n_times // (sp.EPOCH_SEC * fs)
        # 1 Hz rms envelope in the file; MNE holds it at the common 100 Hz, so average per 30-s epoch (uV)
        emg = raw.copy().pick(["submental"]).get_data()[0, : n_ep * sp.EPOCH_SEC * fs].reshape(n_ep, -1).mean(-1) * 1e6
        raw.pick(sp.CHANNELS)
        raw.filter(0.3, 35.0)
        X = raw.get_data()[:, : n_ep * sp.EPOCH_SEC * fs].reshape(len(sp.CHANNELS), n_ep, -1).transpose(1, 0, 2) * 1e6
        _, y, _ = sp.load_recording(psg, hyp)
        # the pipeline's crop (sleep period +- WAKE_MARGIN), recomputed and checked against its labels
        yfull = np.full(n_ep, -1)
        a = mne.read_annotations(hyp)
        for on, du, de in zip(a.onset, a.duration, a.description):
            yfull[int(on // sp.EPOCH_SEC): int((on + du) // sp.EPOCH_SEC)] = sp.STAGE_MAP.get(de, -1)
        asleep = np.where(yfull > 0)[0]
        lo = max(0, asleep[0] - sp.WAKE_MARGIN)
        assert np.array_equal(yfull[lo:lo + len(y)], y), psg
        X, ep = X[lo:lo + len(y)], np.arange(lo, lo + len(y))
        f, P = welch(X, fs=fs, nperseg=4 * fs, axis=-1)          # (epochs, 3 channels, freqs), uV^2/Hz
        tot = band_power(f, P, 0.5, 30)
        r = {"rec": name_of(psg), "epoch": np.arange(len(y)), "stage": y,
             "fpz_ptp_uv": np.ptp(X[:, 0], -1), "fpz_sd_uv": X[:, 0].std(-1), "pz_sd_uv": X[:, 1].std(-1),
             "delta_cover": delta_cover(X[:, 0], fs),
             "eog_slow_uv2": band_power(f, P[:, 2], 0.3, 1.0), "eog_fast_uv2": band_power(f, P[:, 2], 1.0, 5.0),
             "emg_rms_uv": emg[ep]}
        for ch, c in (("fpz", 0), ("pz", 1)):
            for b, (l, h) in {"delta": (0.5, 2), "theta": (4, 8), "alpha": (8, 12), "sigma": (12, 15),
                              "beta": (15, 30)}.items():
                r[f"{ch}_{b}_rel"] = band_power(f, P[:, c], l, h) / tot[:, c]
        rows.append(r)
        print(name_of(psg), len(y), flush=True)
    import pandas as pd
    df = pd.concat([pd.DataFrame(r) for r in rows], ignore_index=True)
    df.to_csv(OUT / "epoch_features.csv", index=False, float_format="%.5g")


# ---------------------------------------------------------------------------------------------------------------
# predict: the fixed pipeline, unchanged, with its test-set answers recorded
# ---------------------------------------------------------------------------------------------------------------
def step_predict(seeds, threads):
    import pandas as pd
    import torch
    torch.set_num_threads(threads)
    orig_load, orig_predict = sp.load_recording, sp.predict
    for seed in seeds:
        path = OUT / f"test_predictions_s{seed}.csv"
        if path.exists():
            continue
        loaded, calls = [], []
        sp.load_recording = lambda psg, hyp: (loaded.append(name_of(psg)), orig_load(psg, hyp))[1]

        def predict(model, ds):
            yt, yp = orig_predict(model, ds)
            calls.append((ds, yt, yp))
            return yt, yp
        sp.predict = predict
        sp.SEED = seed
        sp.main()
        ds, yt, yp = calls[-1]                       # main's last predict() is the single test-set evaluation
        recs = loaded[-len(ds.recs):]                # test recordings are loaded last, in this order
        r, i = np.array(ds.index).T
        pd.DataFrame({"seed": seed, "rec": np.array(recs)[r], "epoch": i + sp.SEQ_LEN // 2, "true": yt, "pred": yp}) \
            .to_csv(path, index=False)
    sp.load_recording, sp.predict = orig_load, orig_predict


# ---------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------
def step_report():
    import pandas as pd
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
    preds = pd.concat([pd.read_csv(p) for p in sorted(OUT.glob("test_predictions_s*.csv"))], ignore_index=True)
    feats = pd.read_csv(OUT / "epoch_features.csv")
    lab = range(5)
    per = []
    for s, g in preds.groupby("seed"):
        for name, fn in (("F1", f1_score), ("recall", recall_score), ("precision", precision_score)):
            per += [{"seed": s, "metric": name, **dict(zip(STAGES, fn(g.true, g.pred, labels=lab, average=None)))}]
        cm = confusion_matrix(g.true, g.pred, labels=lab)
        pd.DataFrame(cm, index=[f"true {c}" for c in STAGES], columns=[f"pred {c}" for c in STAGES]) \
            .to_csv(OUT / f"confusion_s{s}.csv")
    per = pd.DataFrame(per)
    per.to_csv(OUT / "per_stage_metrics.csv", index=False, float_format="%.4f")
    print(per.groupby("metric")[STAGES].agg(["mean", "std"]).round(3).T.to_string())

    cms = np.stack([confusion_matrix(g.true, g.pred, labels=lab) for _, g in preds.groupby("seed")])
    rown = cms / cms.sum(2, keepdims=True)                      # row-normalised: P(pred | true), per seed
    print("\nP(pred | true), mean over seeds (rows true, cols pred):")
    print(pd.DataFrame(rown.mean(0), index=STAGES, columns=STAGES).round(3).to_string())
    print("\nraw counts summed over seeds:")
    print(pd.DataFrame(cms.sum(0), index=STAGES, columns=STAGES).to_string())

    m = preds.merge(feats, on=["rec", "epoch"], how="left", validate="many_to_one")
    assert (m.stage == m.true).all(), "feature epochs misaligned with prediction epochs"
    m.to_csv(OUT / "test_predictions_with_features.csv", index=False, float_format="%.5g")
    cols = ["fpz_ptp_uv", "fpz_sd_uv", "delta_cover", "pz_alpha_rel", "fpz_theta_rel", "fpz_sigma_rel",
            "eog_slow_uv2", "eog_fast_uv2", "emg_rms_uv"]
    print("\nfeature medians by TRUE stage, all 29 recordings:")
    print(feats[feats.stage >= 0].groupby("stage")[cols].median().rename(index=dict(enumerate(STAGES))).round(3).to_string())
    print("\nfeature medians by (true, pred) cell, test windows of all seeds, cells with >= 30 windows:")
    cell = m.groupby(["true", "pred"])
    t = cell[cols].median()
    t.insert(0, "n", cell.size())
    t = t[t.n >= 30].rename(index=dict(enumerate(STAGES)))
    t.round(3).to_csv(OUT / "features_by_cell.csv")
    print(t.round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["facts", "features", "predict", "report"])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    todo = [a.step] if a.step else ["facts", "features", "predict", "report"]
    if "facts" in todo:
        step_facts()
    if "features" in todo and (a.step or not (OUT / "epoch_features.csv").exists()):
        step_features()
    if "predict" in todo:
        step_predict(a.seeds, a.threads)
    if "report" in todo:
        step_report()


if __name__ == "__main__":
    main()
