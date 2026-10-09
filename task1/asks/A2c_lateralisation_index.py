"""A2 (part 3) - Per-subject lateralisation index for mu and beta ERD, with a single-subject statistical test.

Brief: "Define a per-subject lateralisation index for mu (~8-13 Hz) and beta (~13-30 Hz) ERD, with a single-subject
statistical test."

Live part: for SUBJECT, the per-trial C3-C4 ERD difference, the test statistic T and its label-permutation null (10000
permutations, seeded with the subject number exactly as task1/erd.py does, so the p-value is the saved one).
Run: python task1/asks/A2c_lateralisation_index.py
"""
from _common import *

SUBJECT = 72

header("A2.3  Lateralisation index (mu, beta) and single-subject test",
       "define a per-subject lateralisation index for mu and beta ERD, with a single-subject statistical test.",
       "LI = (ipsi - contra) / (|ipsi| + |contra|) of band ERD in dB (> 0: the hemisphere opposite the imagined hand "
       "desynchronises more). Test: T = mean(C3-C4 | left) - mean(C3-C4 | right), one-sided label permutation.")
print_decision("D6")

if has_data():
    data, erd = pipeline("data", "erd")
    D = data.load_subject(SUBJECT)
    X, y, names = D["X"].astype(np.float64), D["y"], list(D["ch_names"])
    times = np.arange(data.N_T) / data.FS + data.TMIN
    P = {ch: erd.power(erd.derive(X, names, ch, "laplacian")) for ch in ("C3", "C4")}
    rng = np.random.default_rng(SUBJECT)        # same seed and draw order as erd.analyse (mu first, then beta)
    saved = csv("a2_lateralisation.csv").set_index("subject").loc[SUBJECT]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    section(f"S{SUBJECT:03d} ({int((y == 0).sum())} left, {int((y == 1).sum())} right trials), Laplacian reference")
    for ax, (band, (lo, hi)) in zip(axes, erd.BANDS.items()):
        e3, e4 = erd.band_erd_db(P["C3"], times, lo, hi), erd.band_erd_db(P["C4"], times, lo, hi)
        contra, ipsi = np.where(y == 0, e4, e3).mean(), np.where(y == 0, e3, e4).mean()
        li = (ipsi - contra) / (abs(ipsi) + abs(contra))
        d = e3 - e4
        T = d[y == 0].mean() - d[y == 1].mean()
        null = np.empty(erd.N_PERM)
        for k in range(erd.N_PERM):
            yp = rng.permutation(y)
            null[k] = d[yp == 0].mean() - d[yp == 1].mean()
        p = (1 + (null >= T).sum()) / (erd.N_PERM + 1)
        print(f"  {band:4s} {lo}-{hi} Hz: contra {contra:+.2f} dB, ipsi {ipsi:+.2f} dB, LI {li:+.3f}, T {T:+.3f} dB, "
              f"p = {p:.4f} ({'significant' if p < erd.ALPHA else 'n.s.'} at {erd.ALPHA} = 0.05/2 bands)")
        check(f"{band} LI", li, saved[f"{band}_LI"])
        check(f"{band} p ", p, saved[f"{band}_p"])
        ax.hist(null, 60, color="0.7")
        ax.axvline(T, color="C3", lw=2, label=f"observed T = {T:.2f} dB, p = {p:.4f}")
        ax.set_title(f"S{SUBJECT:03d} {band}: permutation null of T (10000 label shuffles)", fontsize=9)
        ax.set_xlabel("T = mean(C3-C4 | left) - mean(C3-C4 | right)  [dB]"); ax.legend(fontsize=8)
    print(f"  label: {saved.label}  (present if either band has p < 0.025 AND contralateral ERD < 0 dB)")
    save(fig, f"A2c_permutation_S{SUBJECT:03d}.png")
else:
    no_data_note()

section("All 40 subjects (outputs/a2_lateralisation.csv)")
L = csv("a2_lateralisation.csv")
table(L[["subject", "mu_contra_db", "mu_ipsi_db", "mu_LI", "mu_p", "beta_contra_db", "beta_ipsi_db", "beta_LI", "beta_p", "label"]])
print(f"\n  mean LI: mu {L.mu_LI.mean():+.3f}, beta {L.beta_LI.mean():+.3f};  LI > 0 in {(L.mu_LI > 0).sum()}/40 (mu), "
      f"{(L.beta_LI > 0).sum()}/40 (beta)")

fig, ax = plt.subplots(figsize=(6, 5))
c = np.where(L.label == "present", "C3", "0.5")
ax.scatter(L.mu_LI, L.beta_LI, c=c, s=25)
for _, r in L.iterrows():
    ax.annotate(str(r.subject), (r.mu_LI, r.beta_LI), fontsize=6)
ax.axhline(0, color="k", lw=.4); ax.axvline(0, color="k", lw=.4)
ax.set_xlabel("mu LI"); ax.set_ylabel("beta LI"); ax.set_title("A2.3  LI per subject (red = lateralised ERD present)", fontsize=9)
save(fig, "A2c_LI_all_subjects.png")
finish()
