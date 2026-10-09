"""B1 (part 1) - EEGNet implemented in PyTorch; every hyperparameter changed for this dataset and why.

Brief: "Implement both in PyTorch yourself ... EEGNet (Lawhern et al., 2018, arXiv:1611.08024). Its hyperparameters were
set for a specific sampling rate; state every one you changed for this dataset and why."

Live part: builds task1/models.py EEGNet, runs one batch of the real input shape through it and prints every layer's
output shape and parameter count; then the paper-vs-here hyperparameter table, with the time each kernel spans.
Run: python task1/asks/B1a_eegnet.py
"""
import torch
from _common import *

header("B1.1  EEGNet",
       "implement EEGNet in PyTorch from building blocks; state every hyperparameter changed for this dataset and why.",
       "task1/models.py EEGNet: nn.Conv2d / BatchNorm / AvgPool / Linear only. Paper is written for 128 Hz; this data is 160 Hz.")

models = pipeline("models")
m = models.EEGNet().eval()
x = torch.randn(4, 64, 640)          # batch of 4 trials: 64 electrodes x 640 samples (0-4 s at 160 Hz)

section("Layer by layer on a (4, 64, 640) input")
rows = []
def hook(name):
    return lambda mod, i, o: rows.append((name, type(mod).__name__, tuple(o.shape), sum(p.numel() for p in mod.parameters(recurse=False))))
for name, mod in m.named_modules():
    if name and not list(mod.children()):
        mod.register_forward_hook(hook(name))
with torch.no_grad():
    out = m(x)
for name, kind, shape, n in rows:
    print(f"  {name:12s} {kind:13s} out {str(shape):22s} params {n}")
print(f"  total trainable parameters: {models.n_params(m)}   output: {tuple(out.shape)} (logits left/right)")

section("Hyperparameters: paper (128 Hz) vs here (160 Hz)")
fs = models.FS
H = pd.DataFrame([
    ("F1 temporal filters", 8, m.block1[0].weight.shape[0], "kept"),
    ("D spatial filters per temporal filter", 2, m.spatial.weight.shape[0] // m.block1[0].weight.shape[0], "kept"),
    ("F2 pointwise filters", 16, m.block2[1].weight.shape[0], "kept"),
    ("first kernel (samples)", 64, m.block1[0].weight.shape[-1], f"CHANGED: 0.5 s at {fs} Hz = {m.block1[0].weight.shape[-1]} samples"),
    ("first kernel (seconds)", 64 / 128, m.block1[0].weight.shape[-1] / fs, "same duration -> lowest shaped frequency 2 Hz"),
    ("pool 1", 4, 4, "kept (rate 160 -> 40 Hz)"),
    ("separable kernel (samples)", 16, m.block2[0].weight.shape[-1], "CHANGED: 0.5 s at the pooled 40 Hz"),
    ("separable kernel (seconds)", 16 / 32, m.block2[0].weight.shape[-1] / (fs / 4), "same duration"),
    ("pool 2", 8, 8, "kept (rate 40 -> 5 Hz, 20 time steps into the dense layer)"),
    ("dropout (cross-subject)", 0.25, m.block1[-1].p, "kept"),
    ("max-norm depthwise / dense", "1 / 0.25", "1 / 0.25", "kept (apply_max_norm)"),
], columns=["hyperparameter", "paper", "here", "note"])
print(H.to_string(index=False))
print_decision("D20")
finish()
