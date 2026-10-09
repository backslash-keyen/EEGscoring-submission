"""B2 electrode displacement: test-time cap slides on every saved 30-subject model (DECISIONS D26), plus a layer-by-layer
sensitivity breakdown at the headline 10 mm to locate where each architecture is hurt."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).parent))
import data, train, spatial, b2_run
from models import MODELS

OUT = train.OUT
SHIFTS = [0, 2.5, 5, 10, 15, 20]
DIRS = {"right": (1, 0), "left": (-1, 0), "anterior": (0, 1), "posterior": (0, -1)}


def load_model(jid):
    ck = torch.load(b2_run.RUNS / f"{jid}.pt", weights_only=False)
    r = ck["res"]
    kw = dict(n_ch=21) if r["exp"] == "noconf" else {}
    m = MODELS[r["model"]](**kw)
    m.load_state_dict(ck["state"])
    m.eval()
    norm = train.Norm()
    norm.m, norm.s = ck["m"], ck["s"]
    return m, norm, r


def predict(m, X, bs=512):
    with torch.no_grad():
        return torch.cat([m(torch.from_numpy(X[i:i + bs])) for i in range(0, len(X), bs)]).argmax(1).numpy()


def stages(m, x):
    """Intermediate representations, named by the layer that produced them."""
    out = {}
    with torch.no_grad():
        if hasattr(m, "block1"):                                   # EEGNet
            z = m.block1[1](m.block1[0](x)); out["temporal"] = z
            z = m.block1[2](z); out["spatial (depthwise)"] = z
            z = m.block2(m.block1[3:](z)); out["block2"] = z
        else:
            z = m.temporal(x); out["temporal"] = z
            if m.tokens == "time":
                out["spatial (depthwise)"] = m.spatial(z)
            t = m.embed(x); out["tokens"] = t
            h = torch.cat([m.cls.expand(len(t), -1, -1), t + (m.pos if m.pos is not None else 0)], 1)
            out["cls output"] = m.norm(m.encoder(h)[:, 0])
        out["logits"] = m(x)
    return out


def main():
    torch.set_num_threads(2)   # runs beside the training grid
    names = list(data.load_subject(data.SUBJECTS[0], causal=True)["ch_names"])
    mats = {(s, d): spatial.displacement_matrix(names, s, v).astype(np.float32) for s in SHIFTS for d, v in DIRS.items()}
    jobs = sorted(p.stem for p in b2_run.RUNS.glob("*_n30.json") if p.stem.split("_")[0] in ("base", "aug"))
    CACHE = OUT.parent.parent / "data" / "cache_displacement"   # per-model cache (lets the analysis run while the grid trains); not a result, so it lives in data/
    CACHE.mkdir(exist_ok=True)
    for f in b2_run.FOLDS:
        todo = [j for j in jobs if f"_f{f}_" in j and not (CACHE / f"{j}.csv").exists()]
        if not todo:
            continue
        te = train.split(f)[2]
        X, y, subj, _ = train.load(te)
        for jid in todo:
            rows, layer_rows = [], []
            m, norm, r = load_model(jid)
            for (s, d), M in mats.items():
                if s == 0 and d != "right":
                    continue   # 0 mm is the same in every direction
                Xd = norm(np.einsum("kc,nct->nkt", M, X)).astype(np.float32)
                pred = predict(m, Xd)
                rows.append(dict(job=jid, exp=r["exp"], model=r["model"], fold=f, seed=r["seed"], shift_mm=s,
                                 direction="any" if s == 0 else d, acc=(pred == y).mean(), n=len(y)))
            # where is each model hurt? relative change of each layer's output for a 10 mm slide, first 200 test trials
            x0 = torch.from_numpy(norm(X[:200]).astype(np.float32))
            ref = stages(m, x0)
            for d in DIRS:
                xd = torch.from_numpy(norm(np.einsum("kc,nct->nkt", mats[(10, d)], X[:200])).astype(np.float32))
                for k, v in stages(m, xd).items():
                    layer_rows.append(dict(job=jid, model=r["model"], exp=r["exp"], direction=d, layer=k,
                                           rel_change=((v - ref[k]).norm() / ref[k].norm()).item()))
            pd.DataFrame(layer_rows).to_csv(CACHE / f"{jid}_layers.csv", index=False)
            pd.DataFrame(rows).to_csv(CACHE / f"{jid}.csv", index=False)
            print("done", jid, flush=True)
    files = sorted(CACHE.glob("*.csv"))
    pd.concat([pd.read_csv(p) for p in files if not p.stem.endswith("_layers")]).to_csv(OUT / "b2_displacement_runs.csv", index=False)
    pd.concat([pd.read_csv(p) for p in files if p.stem.endswith("_layers")]).to_csv(OUT / "b2_displacement_layers.csv", index=False)


if __name__ == "__main__":
    main()
