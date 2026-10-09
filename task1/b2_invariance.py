"""B2 tokenisation: what channel tokens WITHOUT an electrode-identity embedding cannot represent, tested directly.
Without identity the encoder is a function of the unordered set of electrode tokens, so f(P x) = f(x) for any channel
permutation P. The left/right mirror (C3<->C4, F7<->F8, ...) is such a permutation, and left- and right-hand imagery are
ideally mirror images, so the model must give a mirrored trial the same answer although its label is the opposite one."""
import re, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).parent))
import data, train, b2_run
from b2_displacement import load_model, predict


def mirror_index(names):
    """Index of each electrode's mirror across the midline: odd number n <-> n+1 (10-10 naming), z-electrodes stay."""
    low = [n.lower() for n in names]
    out = []
    for n in low:
        m = re.match(r"([a-z]+)(\d+)$", n)
        out.append(low.index(f"{m.group(1)}{int(m.group(2)) + (1 if int(m.group(2)) % 2 else -1)}") if m else low.index(n))
    return np.array(out)


def main():
    torch.set_num_threads(4)
    names = list(data.load_subject(data.SUBJECTS[0], causal=True)["ch_names"])
    mir = mirror_index(names)
    rng = np.random.default_rng(0)
    rows = []
    for f in b2_run.FOLDS:
        X, y, _, _ = train.load(train.split(f)[2])
        perm = rng.permutation(64)
        for model in ["tf_chan_noid", "tf_chan_id", "tf_time", "eegnet"]:
            for seed in b2_run.SEEDS:
                jid = f"base_{model}_f{f}_s{seed}_n30"
                if not (b2_run.RUNS / f"{jid}.pt").exists():
                    continue
                m, norm, r = load_model(jid)
                Xn = norm(X).astype(np.float32)
                p0, pm, pp = predict(m, Xn), predict(m, Xn[:, mir]), predict(m, Xn[:, perm])
                rows.append(dict(model=model, fold=f, seed=seed, n=len(y), acc=(p0 == y).mean(),
                                 same_pred_random_perm=(p0 == pp).mean(), same_pred_mirror=(p0 == pm).mean(),
                                 # a mirrored left trial should be called right: accuracy on mirrored data with flipped labels
                                 acc_mirrored_flipped=(pm == 1 - y).mean()))
    D = pd.DataFrame(rows)
    D.to_csv(train.OUT / "b2_invariance_runs.csv", index=False)
    print(D.groupby("model")[["acc", "same_pred_random_perm", "same_pred_mirror", "acc_mirrored_flipped"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
