"""B1 (part 2) - A transformer of my own design, within 3x EEGNet's parameter count.

Brief: "A transformer of your own design, within 3x EEGNet's parameter count." (nn.TransformerEncoderLayer is allowed;
model classes from braindecode/torcheeg are not.)

Live part: builds the three transformer variants used in B2 (time-patch tokens; channel tokens with and without a learned
electrode-identity embedding), counts parameters per component, checks the 3x budget and shows the token shapes.
Run: python task1/asks/B1b_transformer.py
"""
import torch
from _common import *

header("B1.2  Transformer within 3x EEGNet",
       "a transformer of your own design within 3x EEGNet's parameter count.",
       "Shared 0.5 s temporal filter bank (as EEGNet) -> tokens -> 2-layer pre-norm nn.TransformerEncoder (d=16, 2 heads) "
       "-> CLS -> linear. Only the token axis differs between variants.")

models = pipeline("models")
base = models.n_params(models.EEGNet())
budget = 3 * base
x = torch.randn(2, 64, 640)
section(f"Budget: 3 x EEGNet ({base}) = {budget} parameters")
rows = []
for name in ("tf_time", "tf_chan_id", "tf_chan_noid"):
    m = models.MODELS[name]().eval()
    n = models.n_params(m)
    with torch.no_grad():
        tok = m.embed(x)
        out = m(x)
    parts = {k: sum(p.numel() for p in mod.parameters()) for k, mod in m.named_children()}
    parts.update({k: getattr(m, k).numel() for k in ("pos", "cls") if getattr(m, k) is not None})
    rows.append(dict(model=name, params=n, x_eegnet=n / base, within_budget=n <= budget, tokens=tok.shape[1],
                     token_dim=tok.shape[2], output=tuple(out.shape)))
    print(f"\n  {name}: {n} parameters ({n / base:.2f}x EEGNet), {tok.shape[1]} tokens of size {tok.shape[2]}")
    for k, v in parts.items():
        print(f"     {k:10s} {v}")
section("Summary")
table(pd.DataFrame(rows), "{:.2f}")
say("tf_time: token = all electrodes in one 0.2 s segment (spatial filter first, as EEGNet), learned time-position "
    "embedding. tf_chan_*: token = one electrode's filtered time series in 0.5 s segments, no spatial mixing before "
    "attention; with or without a learned electrode-identity embedding (that difference is B2.2).")
print_decision("D21")
finish()
