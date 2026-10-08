"""B1/B2 training harness: cross-subject only. Test fold, validation fold and training pool are disjoint sets of subjects (DECISIONS D22)."""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
import data
from models import MODELS, n_params

OUT = data.ROOT / "outputs" / "partb"
N_FOLDS = 8
T0 = int(-data.TMIN * data.FS)   # sample index of the cue
N_T = 640                        # 0-4.0 s after the cue: the whole imagery period that every regular trial has (D3, D23)


def folds():
    """Same assignment as A3 (D14): fold of each subject from np.random.default_rng(0).permutation, so per-subject
    network accuracies line up with the A3 decoders that were tested on the same held-out subjects."""
    return {int(s): i % N_FOLDS for i, s in enumerate(np.random.default_rng(0).permutation(data.SUBJECTS))}


def split(test_fold, n_train=None, seed=0):
    """Test = fold f, validation (early stopping only) = fold f+1, training pool = the other 30 subjects.
    n_train < 30 draws a seed-dependent subset of the pool, so seeds vary both the subjects and the initialisation."""
    fo = folds()
    test = sorted(s for s, f in fo.items() if f == test_fold)
    val = sorted(s for s, f in fo.items() if f == (test_fold + 1) % N_FOLDS)
    pool = sorted(s for s, f in fo.items() if f not in (test_fold, (test_fold + 1) % N_FOLDS))
    if n_train is not None and n_train < len(pool):
        pool = sorted(np.random.default_rng(1000 * test_fold + seed).choice(pool, n_train, replace=False).tolist())
    return pool, val, test


def load(subjects, win=(T0, T0 + N_T)):
    D = [data.load_subject(s, causal=True) for s in subjects]   # minimum-phase cache: nothing post-cue leaks backwards (D18)
    X = np.concatenate([d["X"][:, :, win[0]:win[1]] for d in D]) * 1e6   # volts -> microvolts
    y = np.concatenate([d["y"] for d in D])
    subj = np.concatenate([np.full(len(d["y"]), s) for s, d in zip(subjects, D)])
    trial = np.concatenate([np.arange(len(d["y"])) for d in D])
    return X.astype(np.float32), y.astype(np.int64), subj, trial


class Norm:
    """Per-channel mean/std from the training subjects only; applied unchanged to validation and test (assignment rule)."""

    def fit(self, X):
        self.m = X.mean((0, 2), keepdims=True)
        self.s = X.std((0, 2), keepdims=True) + 1e-6
        return self

    def __call__(self, X):
        return (X - self.m) / self.s


def shuffle_within(y, subj, rng):
    y = y.copy()
    for s in np.unique(subj):
        m = subj == s
        y[m] = rng.permutation(y[m])
    return y


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def evaluate(model, X, y, bs=256):
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(torch.from_numpy(X[i:i + bs])) for i in range(0, len(X), bs)])
    loss = nn.functional.cross_entropy(logits, torch.from_numpy(y)).item()
    p = logits.softmax(1)[:, 1].numpy()
    return loss, ((p > 0.5) == y).mean(), p


def fit(model_name, test_fold, seed=0, n_train=None, epochs=150, patience=20, lr=1e-3, bs=64, log=print,
        transform=None, model_kw=None, shuffle_labels=False, augment=None, history=None):
    """Train one model on one split. transform(X, subj) -> X lets B2 experiments alter inputs (e.g. remove a confound)
    identically for train/val/test. Returns (model, norm, result dict, per-trial test DataFrame)."""
    tr_s, va_s, te_s = split(test_fold, n_train, seed)
    (Xtr, ytr, str_, _), (Xva, yva, sva, _), (Xte, yte, ste, tte) = load(tr_s), load(va_s), load(te_s)
    if shuffle_labels:   # smoke test of the harness without seeing a real result: labels permuted within subject, expect chance
        rng = np.random.default_rng(12345)
        ytr, yva, yte = (shuffle_within(y, sj, rng) for y, sj in ((ytr, str_), (yva, sva), (yte, ste)))
    if transform is not None:
        Xtr, Xva, Xte = transform(Xtr, tr_s), transform(Xva, va_s), transform(Xte, te_s)
    norm = Norm().fit(Xtr)
    Xraw = Xtr if augment is not None else None   # augmentation acts on the recorded field (microvolts), before normalisation
    Xtr, Xva, Xte = norm(Xtr), norm(Xva), norm(Xte)

    seed_all(seed)
    model = MODELS[model_name](**(model_kw or {}))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    best, best_state, wait, t0 = np.inf, None, 0, time.time()
    for ep in range(epochs):
        model.train()
        tr_loss = []
        for idx in torch.randperm(len(Xtr), generator=gen).split(bs):
            i = idx.numpy()
            xb = Xtr[i] if augment is None else norm(augment(Xraw[i])).astype(np.float32)
            loss = nn.functional.cross_entropy(model(torch.from_numpy(xb)), torch.from_numpy(ytr[i]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            model.apply_max_norm()
            tr_loss.append(loss.item())
        vl, va, _ = evaluate(model, Xva, yva)
        if history is not None:   # per-epoch curve, only for diagnostics
            history.append(dict(epoch=ep, train_loss=float(np.mean(tr_loss)), val_loss=vl, val_acc=va))
        # early stopping on validation-subject loss only; the test fold is never looked at during training
        if vl < best - 1e-4:
            best, best_ep, wait = vl, ep, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    _, val_acc, _ = evaluate(model, Xva, yva)
    _, te_acc, p = evaluate(model, Xte, yte)
    res = dict(model=model_name, fold=test_fold, seed=seed, n_train=len(tr_s), n_params=n_params(model),
               best_epoch=best_ep, epochs_run=ep + 1, val_loss=best, val_acc=val_acc, test_acc=te_acc,
               n_test=len(yte), sec=round(time.time() - t0, 1))
    log(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in res.items()))
    per_trial = pd.DataFrame(dict(model=model_name, fold=test_fold, seed=seed, n_train=len(tr_s), subject=ste,
                                  trial=tte, y=yte, p_right=p))
    return model, norm, res, per_trial


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="eegnet", choices=list(MODELS))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_train", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--shuffle_labels", action="store_true", help="smoke test: permute training labels, expect chance")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    print(pd.DataFrame([(s, f) for s, f in sorted(folds().items())], columns=["subject", "fold"]).groupby("fold").subject.apply(list).to_string())
    fit(a.model, a.fold, a.seed, a.n_train, a.epochs, shuffle_labels=a.shuffle_labels)
