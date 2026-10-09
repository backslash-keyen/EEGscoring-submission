"""A3 (part 2) - Design and run a test for every non-motor route.

Brief: "...then design and run a test for each (restricted-channel, -band or -time decoders are one option)."

Test design: one restricted decoder per route (A3a), L2 logistic regression with C fixed in advance, evaluated
(1) within subject, leave-one-run-out, and (2) pooled across subjects, 8 subject-wise folds. Every decoder has a pre-cue
twin as a method check. All on the minimum-phase (causal) 1-40 Hz cache, the primary run (DECISIONS D18/D19).
Live part: every decoder for SUBJECT (seconds); optionally the pooled F1/M1/R2 decoders over all 40 subjects (LIVE_POOLED).
Run: python task1/asks/A3b_route_tests.py
"""
import contextlib, io
from _common import *

SUBJECT = 72
LIVE_POOLED = True     # recompute three pooled cross-subject accuracies live (~1 min); False = read them from outputs/

header("A3.2  A test for every route",
       "design and run a test for each non-motor route (restricted-channel, -band or -time decoders).",
       "Restricted decoders, leave-one-run-out within subject and 8 subject-wise folds pooled; chance thresholds in A3c.")
print_decision("D13")
print_decision("D14")

W, X_, S = csv("a3_within_subject.csv"), csv("a3_cross_subject.csv"), csv("a3_summary.csv")
if has_data():
    confound, data = pipeline("confound", "data")
    names = list(data.load_subject(SUBJECT, causal=True)["ch_names"])
    ids = list(confound.decoder_specs(names)) + ["R1", "R2", "R12"]
    section(f"Live: every decoder for S{SUBJECT:03d}, leave-one-run-out (permutation part skipped here, it is in the saved run)")
    live = pd.DataFrame(confound.subject_job(SUBJECT, ids, smoke=False, n_perm=0, causal=True))
    mine = W[W.subject == SUBJECT].set_index("decoder").acc
    live["saved_acc"] = live.decoder.map(mine)
    table(live[["decoder", "n", "acc", "saved_acc", "thr_binom", "above_binom"]])
    print(f"  live == saved for {np.isclose(live.acc, live.saved_acc).sum()} / {len(live)} decoders")
    if LIVE_POOLED:
        section("Live: pooled cross-subject accuracy, 8 subject-wise folds, all 40 subjects")
        with contextlib.redirect_stdout(io.StringIO()):   # its progress line would show a 1-permutation p, which means nothing
            xs, _ = confound.cross_subject(["F1", "M1", "R2"], data.SUBJECTS, smoke=False, n_perm=1, causal=True)
        for _, r in xs.iterrows():
            check(f"{r.decoder} pooled acc (n={r.n})", r.acc, X_.set_index("decoder").acc[r.decoder])
else:
    no_data_note()

section("Result per route (primary run; outputs/a3_summary.csv)")
twin = S.set_index("decoder").pooled_xs_acc
rows = []
for route, decs in [("oculomotor", ["F1", "F2"]), ("visual", ["O1", "O2"]), ("muscle", ["G1"]),
                    ("all non-motor sites", ["N2", "N1"]), ("trial order", ["R1", "R2", "R12"]),
                    ("motor (positive control)", ["M1", "A1"])]:
    for d in decs:
        r = S.set_index("decoder").loc[d]
        rows.append(dict(route=route, decoder=d, within_mean=r.mean_acc, subjects_above=f"{int(r.n_above)}/40",
                         p_excess=r.p_excess, pooled_acc=r.pooled_xs_acc, pooled_thr=r.pooled_thr,
                         pre_cue_twin_pooled=twin.get(d + "_pre", np.nan)))
table(pd.DataFrame(rows))
say("subjects_above: how many of 40 subjects beat their own binomial threshold (about 2 expected by luck); p_excess: "
    "probability of that many or more if no subject had a real effect. pre_cue_twin: same decoder before the cue, must be ~0.50.")

sq = csv("a3_sequence.csv")
section("Trial order and run identity")
print(f"  lag-1 same-label pairs: observed {sq.lag1_same.sum()} vs {sq.null_mean.sum():.0f} expected under random order "
      f"(z = {(sq.lag1_same.sum() - sq.null_mean.sum()) / np.sqrt((sq.null_sd ** 2).sum()):.1f})")
print(f"  run-majority oracle (best any run-identity feature can do, analytic): {(sq.oracle_acc * sq.n).sum() / sq.n.sum():.3f}")

section("Follow-up: does the frontal decoder read the current cue or carry-over from the previous trial?")
C = csv("a3_followup_carryover_with_thresholds.csv")
c = C[(C.regime == "pooled_xs") & C.decoder.isin(["F1", "O1", "M1"])]
table(c[["scheme", "decoder", "acc_all", "acc_repeat", "acc_alternate", "thr_repeat", "thr_alternate"]])
say("A decoder that reads carry-over is right on alternating trials and wrong on repeats. F1 is right on both (78% on "
    "repeats): it reads the current cue. 'readout_prev' = trying to decode the PREVIOUS label from the same features.")

section("Method check: the pre-cue twins on the first (zero-phase) vs the primary (minimum-phase) filter")
T = csv("a3c_causal_twins_with_thresholds.csv")
table(T.pivot(index="decoder", columns="filter", values="pooled_xs_acc").reset_index())
say("The zero-phase filter smeared the post-cue eye response up to 1.25 s backwards, so pre-cue windows looked "
    "decodable (F1_pre 61%). With the minimum-phase filter they are at chance (DECISIONS D18).")

show_png("a3_within_subject.png", "A3.2  within-subject accuracy per decoder (one dot per subject)")
show_png("a3c_timecourse.png", "A3.2  pooled accuracy in 100 ms bins: frontal peaks at 0.3-0.4 s (saccade)")
finish()
