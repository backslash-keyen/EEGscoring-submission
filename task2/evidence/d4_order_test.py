"""Defect 4 evidence: does the prediction depend on the ORDER of the neighbouring epochs?
Keep the centre epoch fixed, permute the other 10 epochs of the window, compare outputs."""
import sys, torch
sys.path.insert(0, "..")
from sleep_pipeline import SleepTransformer, SEQ_LEN, set_seed

def run(label):
    set_seed(0)
    m = SleepTransformer(n_ch=3).eval()
    # both orders are written out, so the test does not depend on which one sleep_pipeline.py currently has
    enc = lambda x: m.encoder(x.flatten(0, 1)).view(x.shape[0], x.shape[1], -1)
    if label == "before":  # given script: positions added AFTER the transformer
        m.forward = lambda x: m.head(m.pos(m.transformer(enc(x)))[:, SEQ_LEN // 2])
    else:                  # fix: positions go in BEFORE the transformer
        m.forward = lambda x: m.head(m.transformer(m.pos(enc(x)))[:, SEQ_LEN // 2])
    x = torch.randn(16, SEQ_LEN, 3, 3000); c = SEQ_LEN // 2
    others = [i for i in range(SEQ_LEN) if i != c]
    g = torch.Generator().manual_seed(1)
    diffs = []
    with torch.no_grad():
        base = m(x)
        for _ in range(5):
            perm = [others[i] for i in torch.randperm(len(others), generator=g).tolist()]
            order = list(range(SEQ_LEN))
            for dst, src in zip(others, perm): order[dst] = src
            diffs.append(float((m(x[:, order]) - base).abs().max()))
    print(f"{label:6s} fix: max |output change| after shuffling the 10 neighbours (5 shuffles): "
          + ", ".join(f"{d:.2e}" for d in diffs))

run("before"); run("after")
