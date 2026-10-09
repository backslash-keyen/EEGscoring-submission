"""B2 training grid: one job = (experiment, model, test fold, seed, training-set size). Resumable: a job whose result file
exists is skipped. Every trained model is saved so displacement and attribution analyses need no retraining.
The console log deliberately shows no accuracy (predictions are committed before results are read, PREDICTIONS.md)."""
import argparse, json, sys, time
from pathlib import Path
import multiprocessing as mp
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).parent))
import data, train, confound_free, augment

RUNS = train.OUT / "runs"
FOLDS = [0, 1, 2, 3]          # 20 test subjects, the same for every experiment (DECISIONS D24)
SEEDS = [0, 1, 2]
SIZES = [5, 10, 20, 30]
SCALING_MODELS = ["eegnet", "tf_time"]
TOKEN_MODELS = ["tf_time", "tf_chan_id", "tf_chan_noid"]


def jobs(stage):
    """Priority order, one seed at a time: if the deadline cuts the grid, a whole seed is missing, never a whole experiment.
    Within a seed the 30-subject runs come first (shared by scaling, tokenisation, displacement and attribution)."""
    J = []
    for seed in SEEDS:
        if stage in ("base", "all"):
            for m in ["eegnet"] + TOKEN_MODELS:
                J += [("base", m, f, seed, 30) for f in FOLDS]
            for n in SIZES[:-1]:
                J += [("base", m, f, seed, n) for m in SCALING_MODELS for f in FOLDS]
        if stage in ("extra", "all"):
            J += [("noconf", m, f, seed, 30) for m in SCALING_MODELS for f in FOLDS]
            J += [("aug", "eegnet", f, seed, 30) for f in FOLDS]
    return J


def job_id(j):
    exp, m, f, seed, n = j
    return f"{exp}_{m}_f{f}_s{seed}_n{n}"


def fit_kwargs(j):
    """train.fit arguments of one job; shared with task1/asks so a re-run there trains exactly what the grid trained."""
    exp, m, f, seed, n = j
    kw = dict(test_fold=f, seed=seed, n_train=n, log=lambda s: None,
              # equal number of gradient steps per run at every training-set size, so small sets are not undertrained
              epochs=int(60 * 30 / n), patience=int(10 * 30 / n))
    if exp == "noconf":
        kw.update(transform=confound_free.transform, model_kw=dict(n_ch=len(confound_free.KEEP)))
    if exp == "aug":
        kw.update(augment=augment.Augmenter(seed))
    return kw


def run_job(j, threads):
    torch.set_num_threads(threads)
    exp, m, f, seed, n = j
    jid = job_id(j)
    model, norm, res, per_trial = train.fit(m, **fit_kwargs(j))
    res.update(exp=exp, job=jid)
    torch.save(dict(state=model.state_dict(), m=norm.m, s=norm.s, res=res), RUNS / f"{jid}.pt")
    per_trial.assign(exp=exp).to_csv(RUNS / f"{jid}.csv", index=False)
    (RUNS / f"{jid}.json").write_text(json.dumps(res, default=float))
    return jid, res["epochs_run"], res["sec"]


def _worker(args):
    j, threads = args
    try:
        return run_job(j, threads)
    except Exception as e:   # one failed job must not kill the grid; it is reported and rerun on the next start
        return job_id(j), -1, repr(e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="base", choices=["base", "extra", "all"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs(a.stage) if not (RUNS / f"{job_id(j)}.json").exists()][:a.limit]
    print(f"{len(todo)} jobs to run", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool:
        for k, (jid, ep, sec) in enumerate(pool.imap_unordered(_worker, [(j, a.threads) for j in todo])):
            print(f"[{k + 1}/{len(todo)}] {jid} epochs={ep} sec={sec} elapsed={time.time() - t0:.0f}s", flush=True)
