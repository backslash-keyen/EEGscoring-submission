"""POST-HOC diagnostic (DECISIONS D27): the pre-registered confound-free runs stopped at epoch 0. Does the network fail to
fit the confound-free input at all, or did patience 10 stop it before a weak motor signal emerged? Fold 0, seed 0,
patience 40, per-epoch train/validation curves. The headline confound-free result stays the pre-registered one."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).parent))
import train, confound_free

if __name__ == "__main__":
    torch.set_num_threads(2)
    rows = []
    for model in ["eegnet"]:
        h = []
        m, norm, res, per_trial = train.fit(model, 0, seed=0, n_train=30, epochs=60, patience=40,
                                            transform=confound_free.transform, model_kw=dict(n_ch=21), history=h)
        _, tr_acc, _ = train.evaluate(m, *[a for a in (norm(confound_free.transform(train.load(train.split(0)[0])[0])).astype(np.float32),
                                                         train.load(train.split(0)[0])[1])])
        for r in h:
            rows.append(dict(model=model, **r))
        rows.append(dict(model=model, epoch="final", train_acc_best_ckpt=tr_acc, test_acc=res["test_acc"], best_epoch=res["best_epoch"]))
    pd.DataFrame(rows).to_csv(train.OUT / "b2_noconf_check.csv", index=False)
