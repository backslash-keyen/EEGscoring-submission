"""A2 (part 2) - The undocumented recording reference, and the spatial reference/filter chosen.

Brief: "The dataset does not document the recording reference. Explain what that means for a C3-vs-C4 comparison, then
choose and justify a spatial reference or filter."

Live part: the claim behind the choice, tested on SUBJECT. Every recorded channel is V(site) - V(reference). Re-referencing
the same data to a different single electrode simulates "the reference we were not told about". The raw C3-vs-C4
lateralisation index moves with that choice; the local Laplacian's does not move at all, because its weights sum to zero
and the reference term cancels exactly.
Run: python task1/asks/A2b_reference_choice.py
"""
from _common import *

SUBJECT = 72
FAKE_REFERENCES = ["Cz", "T9", "T10", "Fpz", "Oz"]   # candidate single-electrode references to re-reference to

header("A2.2  Unknown recording reference -> choice of spatial filter",
       "explain what an undocumented reference means for C3 vs C4, then choose and justify a spatial reference or filter.",
       "Compare raw (as recorded), common average (CAR) and local Laplacian (site minus mean of its 4 nearest electrodes); "
       "show by re-referencing that only the Laplacian is independent of the reference.")

section("What the unknown reference means, and the choice (DECISIONS.md)")
print_decision("D4")

if has_data():
    data, erd = pipeline("data", "erd")
    D = data.load_subject(SUBJECT)
    X, y, names = D["X"].astype(np.float64), D["y"], list(D["ch_names"])
    times = np.arange(data.N_T) / data.FS + data.TMIN

    def lat_index(X, ref):
        """mu and beta LI exactly as erd.analyse computes them (same functions), without the permutation test."""
        P = {ch: erd.power(erd.derive(X, names, ch, ref)) for ch in ("C3", "C4")}
        out = {}
        for band, (lo, hi) in erd.BANDS.items():
            e3, e4 = erd.band_erd_db(P["C3"], times, lo, hi), erd.band_erd_db(P["C4"], times, lo, hi)
            c, i = np.where(y == 0, e4, e3).mean(), np.where(y == 0, e3, e4).mean()
            out[band] = (i - c) / (abs(i) + abs(c))
        return out

    section(f"S{SUBJECT:03d}: the three references on the data as recorded (compare with outputs/a2_all_references.csv)")
    saved = csv("a2_all_references.csv")
    for ref in erd.REFS:
        li = lat_index(X, ref)
        s = saved[(saved.subject == SUBJECT) & (saved.ref == ref)].iloc[0]
        print(f"  {ref:9s} mu LI {li['mu']:+.3f}  beta LI {li['beta']:+.3f}   label in pipeline: {s.label}")
        check(f"{ref} mu LI", li["mu"], s.mu_LI)

    section("Re-reference the same recording to other single electrodes (V_site - V_new_ref) and recompute")
    rows = []
    for new_ref in ["as recorded"] + FAKE_REFERENCES:
        Xr = X if new_ref == "as recorded" else X - X[:, [names.index(new_ref)]]
        for ref in ("raw", "laplacian"):
            li = lat_index(Xr, ref)
            rows.append(dict(reference=new_ref, derivation=ref, mu_LI=li["mu"], beta_LI=li["beta"]))
    R = pd.DataFrame(rows)
    table(R.pivot_table(index="reference", columns="derivation", values=["mu_LI", "beta_LI"], sort=False).reset_index())
    spread = R.groupby("derivation")[["mu_LI", "beta_LI"]].agg(lambda v: v.max() - v.min())
    print("\n  range of the LI across the 6 references (max - min):")
    table(spread.reset_index(), "{:.2e}")
    say("Raw C3/C4 power contains the reference electrode's own activity, so the index depends on a choice the dataset "
        "does not report. The Laplacian range is zero up to rounding: the term common to a site and its neighbours cancels.")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    for k, band in enumerate(("mu_LI", "beta_LI")):
        p = R.pivot_table(index="reference", columns="derivation", values=band, sort=False)
        p.plot.bar(ax=ax[k], rot=30)
        ax[k].set_title(f"S{SUBJECT:03d} {band.replace('_', ' ')} under different recording references", fontsize=9)
        ax[k].axhline(0, color="k", lw=.5)
    save(fig, f"A2b_reference_S{SUBJECT:03d}.png")
else:
    no_data_note()

section("All 40 subjects under each derivation (outputs/a2_all_references.csv, written by task1/erd.py)")
A = csv("a2_all_references.csv")
g = A.groupby("ref").agg(present=("label", lambda v: int((v == "present").sum())), mean_mu_LI=("mu_LI", "mean"),
                         mean_beta_LI=("beta_LI", "mean")).reindex(["raw", "car", "laplacian"]).reset_index()
table(g)
lab = A.pivot(index="subject", columns="ref", values="label")
print(f"\n  labels identical across all three derivations for {(lab.nunique(axis=1) == 1).sum()} of 40 subjects")
say("Choice: local Laplacian (primary). It removes any reference, sharpens C3 vs C4 (each site minus its own "
    "neighbourhood) and gives the largest mean LI. CAR also cancels a common reference but spreads frontal eye "
    "artefact into every channel; raw depends on the unknown reference.")
finish()
