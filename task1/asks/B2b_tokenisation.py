"""B2 (part 2) - Tokenisation: time-patch vs channel tokens, and channel tokens with vs without electrode identity.

Brief: "Compare time-patch tokens (all channels, one time segment per token) with channel tokens (one electrode's time
series per token), and channel tokens with vs without a learned electrode-identity embedding. Explain what channel
tokens without identity embedding cannot represent and whether that matters for this task." (Predict first.)

Live part (no training needed): the claim "without identity the model cannot tell electrodes apart" is a property of the
architecture, so it holds for any weights. A freshly initialised model of each variant is given a trial and the same
trial with its electrodes shuffled and mirrored left<->right; only the no-identity model returns identical outputs.
Run: python task1/asks/B2b_tokenisation.py
"""
import torch
from _common import *

SEED = 0      # initialisation of the untrained models in the live test; any seed gives the same yes/no answer

header("B2.2  Tokenisation",
       "time-patch vs channel tokens; channel tokens with vs without electrode identity; what no-identity cannot "
       "represent and whether it matters here.",
       "Three transformers that differ only in the token axis (B1.2), 30 training subjects, 3 seeds, same 20 test subjects.")

section("Prediction (PREDICTIONS.md, B2 item 3)")
say("\n".join(l for l in prediction(r"^## B2 - all six").splitlines() if l.startswith("3.")))

S = csv("partb/b2_summary.csv")
S = S[(S.exp == "base") & (S.n_train == 30)]
L = csv("partb/b2_seed_level.csv")
L = L[(L.exp == "base") & (L.n_train == 30)]
section("Accuracy at 30 training subjects: every seed, then mean +- SD")
print(L.pivot(index="model", columns="seed", values="acc").round(3).to_string())
for _, r in S.iterrows():
    print(f"  {r.model:13s} {100 * r['mean']:.1f} +- {100 * r['std']:.1f}%")

section("Live: is each architecture blind to WHERE a signal is? (untrained models, same input permuted)")
models, b2_inv = pipeline("models", "b2_invariance")
names = list(csv("electrode_positions_2d.csv").name)
mirror = b2_inv.mirror_index(names)
print(f"  mirror map examples: " + ", ".join(f"{names[i]}->{names[mirror[i]]}" for i in [names.index(c) for c in ("C3", "F7", "Fp1", "Cz")]))
if has_data():
    data = pipeline("data")
    D = data.load_subject(72, causal=True)
    x = torch.tensor(D["X"][:8, :, 240:880] * 1e6, dtype=torch.float32)     # 8 real trials, 0-4 s after the cue
    print("  input: 8 real trials of S072 (0-4 s, microvolts)")
else:
    x = torch.randn(8, 64, 640)
    print("  input: random (no data cache found); the property does not depend on the input")
perm = torch.from_numpy(np.random.default_rng(1).permutation(64))
rows = []
for name in ("eegnet", "tf_time", "tf_chan_id", "tf_chan_noid"):
    torch.manual_seed(SEED)
    m = models.MODELS[name]().eval()
    with torch.no_grad():
        y0, yp, ym = m(x), m(x[:, perm]), m(x[:, torch.from_numpy(mirror)])
    rows.append(dict(model=name, max_change_shuffled=float((yp - y0).abs().max()),
                     max_change_mirrored=float((ym - y0).abs().max()),
                     identical_under_shuffle=bool(torch.allclose(yp, y0, atol=1e-5))))
table(pd.DataFrame(rows), "{:.2e}")
say("Without identity, every token is processed by the same weights and attention + CLS read-out sum over tokens, so "
    "the output is a function of the unordered SET of electrode signals. Shuffling or mirroring the electrodes changes "
    "nothing. It cannot represent 'ERD under C3' vs 'ERD under C4', or 'F7 positive, F8 negative' vs the reverse. "
    "(With identity the untrained change is small only because the identity embedding starts near zero, std 0.02; "
    "after training it changes most predictions, see below.)")

section("Same test on the trained models (outputs/partb/b2_invariance_runs.csv, task1/b2_invariance.py)")
I = csv("partb/b2_invariance_runs.csv")
table(I.groupby("model")[["acc", "same_pred_random_perm", "same_pred_mirror", "acc_mirrored_flipped"]].mean().reset_index())
say("acc_mirrored_flipped: accuracy on mirrored trials scored against the flipped label (a left trial mirrored should "
    "look like a right trial). The no-identity model gives exactly 1 - acc: it answers the mirrored trial the same way.")

section("Does it matter for this task?")
say("Yes. The label is itself a left/right mirror variable: both the motor route (contralateral ERD) and the eye route "
    "(saccade toward the target, F7 vs F8 polarity) are mirror patterns. A mirror-invariant model can only use what is "
    "NOT mirror-symmetric in the recording (e.g. an off-centre reference, unequal left/right saccade sizes), hence ~57% "
    "instead of ~70%. Time-patch tokens and channel tokens WITH identity are within seed spread of each other. "
    "Prediction 'time > chan+id = chan-noid': partly wrong (identity is worth ~13 points).")

fig, ax = plt.subplots(figsize=(6, 3.8))
order = ["eegnet", "tf_time", "tf_chan_id", "tf_chan_noid"]
for k, m in enumerate(order):
    a = 100 * L[L.model == m].acc.values
    ax.scatter(np.full(len(a), k), a, color=f"C{k}")
    ax.hlines(a.mean(), k - .25, k + .25, color="k")
ax.axhline(100 * S.thr_binom_5pct.iloc[0], color="0.5", ls=":")
ax.set_xticks(range(4)); ax.set_xticklabels(order); ax.set_ylabel("test accuracy (%), dots = seeds")
ax.set_title("B2.2  tokenisation at 30 training subjects (dotted = chance threshold)", fontsize=9)
save(fig, "B2b_tokenisation.png")
finish()
