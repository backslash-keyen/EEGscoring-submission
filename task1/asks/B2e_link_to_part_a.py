"""B2 (part 5) - Link to Part A: per-subject correlations, and a retrain with the dominant confound removed.

Brief: "Per test subject, correlate accuracy with the A2 lateralisation index and the A3 confound decoders. Retrain with
the dominant confound removed (your method) and report what remains." (Predict first.)

Live part: recomputes the Spearman correlations from the per-subject table and checks them against the saved ones;
shows on synthetic scalp fields why the removal method (Laplacian, then sensorimotor electrodes only) cancels a broad
eye field but keeps a focal C3 source.
Run: python task1/asks/B2e_link_to_part_a.py
"""
from scipy.stats import spearmanr
from _common import *

header("B2.5  Link to Part A and confound removal",
       "per test subject, correlate network accuracy with the A2 LI and the A3 confound decoders; retrain with the "
       "dominant confound removed and report what remains.",
       "20 test subjects, accuracy = mean over 3 seeds at 30 training subjects; Spearman. Dominant confound (A3) = eye "
       "movement. Removal = local Laplacian over all 64 electrodes, then the 21 Fc/C/Cp electrodes only.")

section("Prediction (PREDICTIONS.md, B2 item 7)")
say("\n".join(l for l in prediction(r"^## B2 - all six").splitlines() if l.startswith("7.")))

P = csv("partb/b2_link_per_subject.csv")
C = csv("partb/b2_link_correlations.csv")
section("Spearman rho (p) of per-subject accuracy vs Part A measures, recomputed now and checked against the saved table")
rows = []
for (exp, model), g in P.groupby(["exp", "model"]):
    row = dict(exp=exp, model=model)
    for meas in ("LI_mean", "acc_F1", "acc_M1", "acc_O1"):
        rho, p = spearmanr(g.net_acc, g[meas])
        saved = C[(C.exp == exp) & (C.model == model) & (C.part_a_measure == meas)].spearman_rho.iloc[0]
        assert abs(rho - saved) < 1e-9, (exp, model, meas)
        row[meas] = f"{rho:+.2f} (p={p:.3f})"
    rows.append(row)
print(pd.DataFrame(rows).rename(columns={"LI_mean": "A2 LI (mu,beta mean)", "acc_F1": "A3 F1 frontal eye",
                                         "acc_M1": "A3 M1 motor strip", "acc_O1": "A3 O1 occipital"}).to_string(index=False))
print("  all values equal outputs/partb/b2_link_correlations.csv")
a = csv("a3_per_subject_table.csv")
rho, p = spearmanr((a.mu_LI + a.beta_LI) / 2, a.acc_F1)
print(f"  across all 40 subjects, A2 LI vs A3 F1: rho = {rho:+.2f} (p = {p:.2f}) -> the two routes are unrelated, so an "
      "accuracy that tracks both draws on both, in different subjects")

g = P[(P.exp == "base") & (P.model == "eegnet")]
fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for k, (meas, lab) in enumerate((("LI_mean", "A2 lateralisation index (mean of mu, beta)"), ("acc_F1", "A3 frontal eye decoder accuracy"))):
    ax[k].scatter(g[meas], 100 * g.net_acc, c=np.where(g.label == "present", "C3", "0.5"))
    for _, r in g.iterrows():
        ax[k].annotate(str(r.subject), (r[meas], 100 * r.net_acc), fontsize=6)
    ax[k].set_xlabel(lab); ax[k].set_title(f"rho = {spearmanr(g.net_acc, g[meas])[0]:+.2f}", fontsize=9)
ax[0].set_ylabel("EEGNet test accuracy (%), per test subject")
fig.suptitle("B2.5  per-subject EEGNet accuracy vs Part A (red = lateralised ERD present)", fontsize=10)
save(fig, "B2e_link_scatter.png")

section("Why the removal method removes the eye field but keeps motor ERD (live, task1/confound_free.py)")
if has_data():   # confound_free reads the channel order from the data cache
    confound_free, spatial = pipeline("confound_free", "spatial")
    W = confound_free.weights()                                  # (21, 64): Laplacian rows of the 21 kept electrodes
    names = list(csv("electrode_positions_2d.csv").name)
    pos, _ = spatial.positions(names)
    fields = {"constant (any reference term)": np.ones(64),
              "broad left-right gradient (horizontal EOG far field)": pos[:, 0] / np.abs(pos[:, 0]).max(),
              "broad front-back gradient (vertical EOG far field)": pos[:, 1] / np.abs(pos[:, 1]).max(),
              "focal source under C3 (motor ERD)": np.exp(-np.linalg.norm(pos - pos[names.index("C3")], axis=1) ** 2 / (2 * 0.025 ** 2))}
    keep = [names.index(c) for c in confound_free.KEEP]
    print(f"  Laplacian row sums (must be 0): max |sum| = {np.abs(W.sum(1)).max():.1e}")
    for lab, f in fields.items():
        print(f"  {lab:52s} kept after removal: {100 * np.linalg.norm(W @ f) / np.linalg.norm(f[keep]):5.1f}% of its size at those electrodes")
else:
    no_data_note()
print_decision("D25")

section("Retrained with the eye route removed: what remains (every seed, 30 training subjects)")
L = csv("partb/b2_seed_level.csv")
L = L[(L.n_train == 30) & L.model.isin(["eegnet", "tf_time"]) & L.exp.isin(["base", "noconf"])]
print((100 * L.pivot_table(index=["model", "exp"], columns="seed", values="acc")).round(1).to_string())
S = csv("partb/b2_summary.csv")
for _, r in S[(S.exp == "noconf")].iterrows():
    print(f"  {r.model}: {100 * r['mean']:.1f} +- {100 * r['std']:.1f}% (chance threshold {100 * r.thr_binom_5pct:.1f}%)")

section("Post-hoc diagnostic: does the network fail to fit, or fail to transfer? (DECISIONS D27)")
N = csv("partb/b2_noconf_check.csv")
ep = N[N.epoch != "final"].astype({"epoch": int})
print(f"  epoch 0 -> {ep.epoch.max()}: training loss {ep.train_loss.iloc[0]:.2f} -> {ep.train_loss.iloc[-1]:.2f}, validation-subject "
      f"loss {ep.val_loss.iloc[0]:.2f} -> {ep.val_loss.iloc[-1]:.2f}, validation accuracy max {100 * ep.val_acc.max():.1f}%")
say("It fits the training subjects' motor data but what it learns does not transfer to new subjects. What remains "
    "cross-subject: essentially nothing. The motor signal exists (A2: 14/40 subjects lateralised) but its strength and "
    "topography are subject-specific; the eye signal was the only pattern shared across subjects. Prediction 'A2 LI; "
    "small drop': correlation part right (EEGNet tracks LI most), drop part wrong (to chance).")
finish()
