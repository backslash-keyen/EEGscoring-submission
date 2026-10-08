"""B2 what the model relies on: attention and gradient x input for the best transformer, faithfulness by token deletion,
and an input-space map (electrodes x time) for comparison with A2 (C3/C4 mu/beta) and A3 (frontal 0-0.5 s)."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne

sys.path.insert(0, str(Path(__file__).parent))
import data, train, b2_run, confound
from b2_displacement import load_model

OUT = train.OUT
FRACS = [0.05, 0.1, 0.2, 0.3, 0.5]
N_RANDOM = 5
REGIONS = {"frontal (A3 eye)": confound.FRONT, "occipital (A3 visual)": confound.OCC, "temporal": confound.TEMP,
           "sensorimotor (A2)": confound.SENSORIMOTOR}


def best_transformer():
    """Highest mean test accuracy over folds 0-3 x seeds at 30 training subjects; rule fixed before results (D24)."""
    R = pd.DataFrame([json.loads(p.read_text()) for p in b2_run.RUNS.glob("base_tf_*_n30.json")])
    acc = R.groupby("model").test_acc.mean()
    return acc.idxmax(), acc


def tokens_in(m, x):
    t = m.embed(x)
    return t + m.pos if m.pos is not None else t


def from_tokens(m, t):
    h = torch.cat([m.cls.expand(len(t), -1, -1), t], 1)
    return m.head(m.norm(m.encoder(h)[:, 0]))


def cls_attention(m, t):
    """Attention from CLS to each token in the last layer, averaged over heads (what the read-out position collects)."""
    h = torch.cat([m.cls.expand(len(t), -1, -1), t], 1)
    for layer in m.encoder.layers[:-1]:
        h = layer(h)
    last = m.encoder.layers[-1]
    z = last.norm1(h)   # pre-norm layer: attention sees the normalised input
    _, a = last.self_attn(z, z, z, need_weights=True, average_attn_weights=True)
    return a[:, 0, 1:]


def grad_x_input(m, t, y_pred):
    """|sum_d t_d * d(logit_pred - logit_other)/d t_d| per token: first-order effect of removing the token's embedding."""
    t = t.detach().requires_grad_(True)
    out = from_tokens(m, t)
    sign = torch.where(torch.from_numpy(y_pred) == 1, 1.0, -1.0)
    (sign * (out[:, 1] - out[:, 0])).sum().backward()
    return (t.grad * t).sum(-1).abs().detach()


def input_map(m, x, y_pred):
    """grad x input on the raw (normalised) EEG: (trials, channels, time) importance, for the electrode / time maps."""
    x = x.clone().requires_grad_(True)
    out = m(x)
    sign = torch.where(torch.from_numpy(y_pred) == 1, 1.0, -1.0)
    (sign * (out[:, 1] - out[:, 0])).sum().backward()
    return (x.grad * x).abs().detach()


def accuracy_with_mask(m, x, y, mask, bs=256):
    with torch.no_grad():
        p = torch.cat([m(x[i:i + bs], token_mask=mask[i:i + bs]) for i in range(0, len(x), bs)]).argmax(1).numpy()
    return (p == y).mean()


def main():
    torch.set_num_threads(4)
    best, acc = best_transformer()
    print("transformer test accuracy by variant:", acc.round(4).to_dict(), "-> best:", best, flush=True)
    names = list(data.load_subject(data.SUBJECTS[0], causal=True)["ch_names"])
    tok_rows, faith_rows, imaps = [], [], {}
    for f in b2_run.FOLDS:
        X, y, subj, _ = train.load(train.split(f)[2])
        for model_name in (best, "eegnet"):
            for seed in b2_run.SEEDS:
                jid = f"base_{model_name}_f{f}_s{seed}_n30"
                m, norm, r = load_model(jid)
                x = torch.from_numpy(norm(X).astype(np.float32))
                with torch.no_grad():
                    y_pred = torch.cat([m(x[i:i + 256]) for i in range(0, len(x), 256)]).argmax(1).numpy()
                im = torch.cat([input_map(m, x[i:i + 128], y_pred[i:i + 128]) for i in range(0, len(x), 128)])
                imaps.setdefault(model_name, []).append(im.mean(0).numpy())     # (64, 640)
                if model_name == "eegnet":
                    continue
                t = torch.cat([tokens_in(m, x[i:i + 256]).detach() for i in range(0, len(x), 256)])
                with torch.no_grad():
                    att = torch.cat([cls_attention(m, t[i:i + 256]) for i in range(0, len(t), 256)])
                g = torch.cat([grad_x_input(m, t[i:i + 128], y_pred[i:i + 128]) for i in range(0, len(t), 128)])
                labels = names if m.tokens == "channel" else [f"{0.2 * k:.1f}-{0.2 * (k + 1):.1f}s" for k in range(t.shape[1])]
                for k, lab in enumerate(labels):
                    tok_rows.append(dict(job=jid, token=lab, attention=att[:, k].mean().item(), gradxinput=g[:, k].mean().item()))
                # faithfulness: delete each trial's top-k tokens by each score (key-padding mask + zeroed embedding)
                N = t.shape[1]
                rng = np.random.default_rng(100 * f + seed)
                faith_rows.append(dict(job=jid, method="none", frac=0.0, acc=accuracy_with_mask(m, x, y, torch.zeros(len(x), N, dtype=torch.bool))))
                for frac in FRACS:
                    k = max(1, int(round(frac * N)))
                    for method, score in (("attention", att), ("gradxinput", g)):
                        top = score.argsort(1, descending=True)[:, :k]
                        mask = torch.zeros(len(x), N, dtype=torch.bool).scatter_(1, top, True)
                        faith_rows.append(dict(job=jid, method=method, frac=frac, acc=accuracy_with_mask(m, x, y, mask)))
                    for rep in range(N_RANDOM):
                        top = torch.from_numpy(np.argsort(rng.random((len(x), N)), 1)[:, :k])
                        mask = torch.zeros(len(x), N, dtype=torch.bool).scatter_(1, top, True)
                        faith_rows.append(dict(job=jid, method="random", frac=frac, rep=rep, acc=accuracy_with_mask(m, x, y, mask)))
                print("done", jid, flush=True)
    pd.DataFrame(tok_rows).to_csv(OUT / "b2_attr_tokens.csv", index=False)
    F = pd.DataFrame(faith_rows)
    F.to_csv(OUT / "b2_faithfulness_runs.csv", index=False)

    # input-space maps: per electrode (summed over time) and per 0.1 s bin (summed over electrodes), normalised to 1
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    info = mne.create_info(names, data.FS, "eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1005"), match_case=False)
    for col, (model_name, maps) in enumerate(imaps.items()):
        A = np.mean(maps, 0)
        ch = A.sum(1) / A.sum()
        tb = A.reshape(64, 40, 16).sum((0, 2)) / A.sum()
        for c, v in zip(names, ch):
            rows.append(dict(model=model_name, kind="electrode", key=c, share=v))
        for k, v in enumerate(tb):
            rows.append(dict(model=model_name, kind="time_0.1s", key=round(0.1 * k, 1), share=v))
        for reg, chans in REGIONS.items():
            rows.append(dict(model=model_name, kind="region", key=reg, share=ch[confound.idx(names, chans)].sum(),
                             n_electrodes=len(chans)))
        mne.viz.plot_topomap(ch, info, axes=axes[0, col], show=False, contours=0)
        axes[0, col].set_title(f"{model_name}: grad x input by electrode")
        axes[1, col].bar(np.arange(40) * 0.1 + 0.05, tb, width=0.09)
        axes[1, col].axvspan(0, 0.5, color="tab:red", alpha=0.1, label="A3 eye window 0-0.5 s")
        axes[1, col].set(xlabel="time after cue (s)", ylabel="share of attribution", title=f"{model_name}: by time")
        axes[1, col].legend()
    fig.tight_layout()
    fig.savefig(OUT / "b2_attribution_maps.png", dpi=120)
    pd.DataFrame(rows).to_csv(OUT / "b2_attr_input.csv", index=False)

    s = F[F.method != "none"].groupby(["method", "frac"]).acc.agg(["mean", "std"]).reset_index()
    s.to_csv(OUT / "b2_faithfulness.csv", index=False)
    fig, ax = plt.subplots(figsize=(5, 4))
    base = F[F.method == "none"].acc.mean()
    for method, g in s.groupby("method"):
        ax.errorbar([0] + list(g.frac), [base] + list(g["mean"]), yerr=[0] + list(g["std"]), marker="o", label=method, capsize=3)
    ax.set(xlabel="fraction of tokens deleted (top-ranked per trial)", ylabel="test accuracy", title=f"faithfulness: {best}")
    ax.axhline(0.5, color="grey", lw=0.8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "b2_faithfulness.png", dpi=120)


if __name__ == "__main__":
    main()
