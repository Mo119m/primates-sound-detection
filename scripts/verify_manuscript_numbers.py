"""Recompute the manuscript's numeric claims from their source files.

Today one of my own stated numbers -- "146 windows unique to blocks 3+4, 134
unheard" -- turned out to be 103 and 97 once it was recomputed with a sane
matching tolerance instead of exact start-time equality. It had been quoted in
conversation and never checked against a file. The manuscript has a longer list
of numbers with the same exposure, so this walks them.

Every check prints the claimed value, the recomputed value, and OK/OFF. A claim
whose source file is missing prints SKIP with the reason rather than passing
silently -- "no file" and "matches" must not look alike.
"""
import os
import re

import pandas as pd

# Relative to this file, not an absolute path: the version of this that
# lived in a scratch directory hard-coded one developer's drive, which is
# fine for a throwaway and wrong for a check the paper depends on.
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TEX = os.path.join(REPO, "overleaf/methodsx_manuscript.tex")
REVIEW = os.path.join(REPO, "data/outputs/auto_cleanup/cleanup_vs_review.csv")

RESULTS = []


def check(name, claimed, actual, tol=0.0):
    if actual is None:
        RESULTS.append(("SKIP", name, claimed, "source missing"))
        return
    try:
        good = abs(float(claimed) - float(actual)) <= tol
    except (TypeError, ValueError):
        good = str(claimed) == str(actual)
    RESULTS.append(("OK" if good else "OFF", name, claimed, actual))


def maybe(path):
    p = os.path.join(REPO, path)
    return pd.read_csv(p) if os.path.exists(p) else None


def main():
    tex = open(TEX, encoding="utf-8").read()
    rev = pd.read_csv(REVIEW, dtype=str, keep_default_na=False)

    # ---- the review itself, which everything else is scored against ----
    check("review rows (abstract: 6,189 reviewed detections)", 6189, len(rev))
    check("confirmed calls (limitation 5: 2,535)", 2535,
          int((rev["verdict"] == "call").sum()))
    check("false positives (3,654 manual_id=Noise rows)", 3654,
          int((rev["verdict"] == "false_positive").sum()))

    # ---- Table 2, the main LOSO sweep ----
    #
    # NOT v13_loso_final.csv. That file was assumed to be the source for weeks
    # and is not: testing all 4,711 CSVs under data/outputs row by row against
    # the sixteen printed rows, on all six columns, matches exactly one file,
    # 16/16 to the last printed decimal. The table also uses that file's
    # *time-gated* columns, which is why the plain ones disagree.
    # As of 2026-08-25 the table is the sixteen-fold sweep on the finished
    # 22,169-row dataset. It was the all-background sweep from 2026-08-18, and
    # the armA one before that, and the file below went on naming armA for four
    # days after the table changed -- during which every check in this section
    # passed, because each compares a constant written here against the CSV
    # named here and neither had anything to do with the manuscript. That is
    # worse than no check: a stale assertion that cannot fail reads as coverage.
    #
    # The tex_check guard added afterwards is what caught the 2026-08-25 move:
    # thirteen checks went OFF the moment the manuscript changed and this file
    # had not, which is the wanted behaviour. It does not catch everything.
    # Six numbers in the prose -- the abstract's review-set size, IPA4ST's
    # fitted threshold and gated precision, IPA20ST's deployed precision, and
    # both ends of the matched-threshold range -- were found on 2026-08-25 to
    # reproduce uniquely from armA, two runs stale, while the table beside them
    # named a different run entirely. Nothing here was watching them. They are
    # pinned below now, under "prose numbers".
    t = maybe("data/outputs/v13_runs/full_2026-08-19/loso16_freqpos_evalfix.csv")
    old = maybe("data/outputs/v13_runs/armA_corrections/loso16.csv")
    if t is not None:
        # Every constant below is also asserted to appear in the .tex. That is
        # the guard this section did not have: a number can otherwise be
        # updated in the manuscript, left alone here, and go on passing against
        # whichever CSV happens to be named. Present-in-the-text is a weak
        # check on its own -- it does not know which sentence -- but it is
        # enough to make silent divergence impossible, and the cell-by-cell
        # table comparison covers the rest.
        def tex_check(name, printed, actual, tol=None):
            if printed not in tex:
                check(f"{name} -- '{printed}' is not in the manuscript",
                      True, False)
                return
            # The manuscript writes thousands with a LaTeX thin space, 6\,478,
            # and some numbers with a brace group, 6{,}478. Both are the same
            # number; strip the separator rather than making every caller
            # pass a second, plain copy of what it already wrote.
            want = float(printed.replace("\\,", "").replace("{,}", ""))
            check(name, want, actual, tol if tol is not None else 0)

        st = lambda s, c: float(t.loc[t["station"] == s, c].iloc[0])
        check("Table 2 stations", 16, len(t))
        check("gated detections total (3,497)", 3497,
              int(t["gated_detections"].sum()))
        tex_check("macro v12_precision (abstract)", "0.711",
                  round(t["gated_v12_precision"].mean(), 3), 0.001)
        tex_check("macro loso_precision (abstract)", "0.960",
                  round(t["gated_loso_precision"].mean(), 3), 0.001)
        tex_check("macro fps removed %", "90.4",
                  round(100 * t["gated_loso_fps_removed"].mean(), 1), 0.1)
        tex_check("macro calls retained %", "92.0",
                  round(100 * t["gated_loso_calls_retained"].mean(), 1), 0.1)
        tex_check("weakest fold IPA7ST", "0.873",
                  round(st("IPA7ST", "gated_loso_precision"), 3), 0.001)
        tex_check("worst recall IPA7ST", "79.7",
                  round(100 * st("IPA7ST", "gated_loso_calls_retained"), 1), 0.1)
        tex_check("second worst recall IPA13ST", "82.6",
                  round(100 * st("IPA13ST", "gated_loso_calls_retained"), 1), 0.1)
        # The stations that remove every false positive. Three of them until
        # 2026-08-18, four on the all-background sweep, two on this one: it is
        # a property of where each fold's fitted threshold lands and not a
        # stable fact about any station, so the count is read from the file
        # rather than asserted, and only the agreement with the .tex is fixed.
        perfect = sorted(t.loc[t["gated_loso_fps_removed"] >= 1.0, "station"])
        check("stations removing every false positive (two)", 2, len(perfect))
        for s in perfect:
            check(f"  {s} named in that sentence", True,
                  s in tex.split("remove every false positive")[0][-200:]
                  if "remove every false positive" in tex else False)
        # The four checks that used to sit here compared this table against the
        # armA one and asserted the manuscript's account of the difference:
        # macro 0.921, calls retained 95.0 %, twelve of sixteen folds improved.
        # That paragraph was replaced on 2026-08-18 by the BirdNET null, and
        # none of those numbers appears in the .tex any more, so the checks were
        # asserting the contents of deleted prose. Removed rather than updated:
        # a check whose subject no longer exists has nothing to be corrected to.

        # threshold spread, as the manuscript now states it
        th = t["gated_loso_threshold"]
        tex_check("threshold median", "0.806",
                  round(float(th.median()), 3), 0.001)
        tex_check("threshold Q1", "0.627",
                  round(float(th.quantile(.25)), 3), 0.001)
        tex_check("threshold Q3", "0.908",
                  round(float(th.quantile(.75)), 3), 0.001)
        tex_check("threshold min", "0.238", round(float(th.min()), 3), 0.001)
        tex_check("threshold max", "0.946", round(float(th.max()), 3), 0.001)
        check("thresholds below 0.9 (ten)", 10, int((th < 0.9).sum()))
        check("thresholds below 0.5 (two)", 2, int((th < 0.5).sum()))
        check("manuscript says thirteen below 0.9", True,
              "ten of sixteen folds fall" in tex)
        check("manuscript says two below 0.5", True, "two below 0.5" in tex)

        # ---- prose numbers ----
        #
        # Six figures that live in sentences rather than in the table, and that
        # nothing was watching. On 2026-08-25 every one of them was found to
        # reproduce uniquely from armA_corrections -- two runs behind the table
        # printed beside them -- which is how a paper ends up quoting a
        # threshold of 0.964 on the same page as a table saying 0.451.
        #
        # These are tex_checks, so they fail in both directions: if the run
        # changes and the sentence does not, or if the sentence is edited to a
        # value the run does not support.
        mt = t["gated_matched_threshold"]
        tex_check("abstract review-set size", "6\\,110",
                  int(t["detections"].sum()))
        tex_check("IPA4ST fitted threshold in prose", "0.844",
                  round(st("IPA4ST", "gated_loso_threshold"), 3), 0.001)
        tex_check("IPA4ST gated precision in prose", "0.978",
                  round(st("IPA4ST", "gated_loso_precision"), 3), 0.001)
        tex_check("IPA20ST deployed precision in prose", "0.954",
                  round(st("IPA20ST", "gated_v12_precision"), 3), 0.001)
        tex_check("matched-threshold floor", "0.0021",
                  round(float(mt.min()), 4), 0.0001)
        tex_check("matched-threshold ceiling", "0.9775",
                  round(float(mt.max()), 4), 0.0001)
        # The two stations the scanning check was run at are no longer the
        # extremes of the table. The manuscript has to say so rather than
        # repeat the claim; this pins the admission, not the old claim.
        lo = t.loc[t["gated_v12_precision"].idxmin(), "station"]
        hi = t.loc[t["gated_v12_precision"].idxmax(), "station"]
        check("deployed-precision extremes are IPA1ST/IPA20ST",
              ("IPA1ST", "IPA20ST"), (lo, hi))
        # The scanning check was run at IPA20ST and IPA4ST as the extremes of
        # the table as it then stood. On the corrected run IPA20ST is still
        # the upper extreme and IPA1ST has displaced IPA4ST as the lower; the
        # manuscript must say so rather than repeat the original claim.
        check("manuscript places the lower extreme at IPA1ST",
              True, "the lower is now" in tex and "IPA1ST at 0.438" in tex)
    else:
        for n in ("Table 2", "abstract macros", "threshold spread"):
            check(n, "-", None)

    # ---- the printed rows of Table 2, parsed out of the .tex ----
    #
    # The macros above can all pass while a single cell is mistyped, and the
    # table was retyped by hand today. So each printed row is parsed and
    # compared against the CSV it claims to report.
    if t is not None:
        body = tex.split(r"\label{tab:loso}")[1].split(r"\bottomrule")[0]
        printed = {}
        for line in body.splitlines():
            line = line.strip()
            if not line.startswith("IPA"):
                continue
            cells = [c.strip() for c in line.rstrip("\\").split("&")]
            if len(cells) == 7:
                printed[cells[0]] = cells[1:]
        check("Table 2 printed rows", 16, len(printed))
        cols = ["gated_detections", "gated_v12_precision", "gated_loso_threshold",
                "gated_loso_precision", "gated_loso_fps_removed",
                "gated_loso_calls_retained"]
        bad = []
        for st, cells in printed.items():
            row = t[t["station"] == st]
            if row.empty:
                bad.append(f"{st}: not in the CSV")
                continue
            row = row.iloc[0]
            want = [f"{int(row[cols[0]])}", f"{row[cols[1]]:.3f}",
                    f"{row[cols[2]]:.3f}", f"{row[cols[3]]:.3f}",
                    f"{100 * row[cols[4]]:.1f}", f"{100 * row[cols[5]]:.1f}"]
            for i, (got, exp) in enumerate(zip(cells, want)):
                if got != exp:
                    bad.append(f"{st} col{i + 2}: printed {got}, file {exp}")
        check("Table 2 cells matching the CSV", 0, len(bad))
        for b in bad[:10]:
            print("   !", b)

    # The six-fold head ablation used to be checked here: means of 0.822, 0.792
    # and 0.854, a paired mean of +0.032 and an sd of 0.240. Those five checks
    # passed to the end, against the armA run they name, after the sixteen-fold
    # ablation replaced that section on 2026-08-19 and none of the five numbers
    # remained anywhere in the .tex. Removed rather than repointed: the claims
    # they verified no longer exist, and a check with no subject is the failure
    # mode this file was rewritten to stop making. The replacement checks are
    # below, under the sixteen-fold ablation. Its per-fold companions -- six
    # folds compared, four positive, +0.433 and -0.321 at the extremes, and the
    # retained/removed pairs 0.938/0.934 and 0.693/0.798 -- went with it, for
    # the same reason and in the same edit.

    # ---- the three unfreeze runs ----
    cu = "data/outputs/v13_runs/colab_unfreeze"
    for f, claimed in (("out_frozen.csv", 0.169), ("out_block4.csv", 0.889),
                       ("out_block34.csv", 0.979)):
        d = maybe(f"{cu}/{f}")
        v = None if d is None else round(
            float(d.loc[d["station"] == "IPA4ST", "loso_precision"].iloc[0]), 3)
        check(f"{f} IPA4ST held-out precision", claimed, v, 0.001)
    # armA's own three-fold run, named here rather than inherited from the
    # deleted block above, which is where `arm` used to come from.
    a = maybe("data/outputs/v13_runs/armA_corrections/loso.csv")
    if a is not None:
        check("armA IPA4ST held-out precision (0.962)", 0.962,
              round(float(a.loc[a["station"] == "IPA4ST", "loso_precision"].iloc[0]), 3),
              0.001)
        check("armA macro over three folds (0.965)", 0.965,
              round(float(a["loso_precision"].mean()), 3), 0.001)

    # ---- IPA4ST's instability across repeats of one specification ----
    a3 = maybe("data/outputs/v13_runs/armA_corrections/loso.csv")
    a16 = maybe("data/outputs/v13_runs/armA_corrections/loso16.csv")
    if a3 is not None and a16 is not None:
        def at(d, st, col="loso_precision"):
            return round(float(d.loc[d["station"] == st, col].iloc[0]), 4)
        check("IPA4ST draw 1, laptop 3-fold (0.962)", 0.962, at(a3, "IPA4ST"), 0.001)
        check("IPA4ST draw 2, laptop 16-fold (0.924)", 0.924, at(a16, "IPA4ST"), 0.001)
        # draw 3 is the Colab run; its CSV reaches Drive only after the second
        # arm finishes, so it is checked against the file once fetched. Until
        # then this stays a SKIP rather than a literal, because a number that
        # lives only in a transcript is the failure this file exists to prevent.
        cf = maybe("data/outputs/v13_runs/colab_armA/out_armA_frozen.csv")
        check("IPA4ST draw 3, Colab T4 (0.077)", 0.077,
              at(cf, "IPA4ST") if cf is not None else None, 0.001)
        for st, v1, v2 in (("IPA20ST", 0.9684, 0.9684), ("IPA13ST", 0.9649, 0.9643)):
            check(f"{st} stable across draws 1-2 ({v1})", v1, at(a3, st), 0.0001)
            check(f"{st} stable across draws 1-2 ({v2})", v2, at(a16, st), 0.0001)
        # the base-rate explanation
        br = a16.set_index("station")
        check("IPA4ST base rate 4.0%", 4.0,
              round(100 * br.loc["IPA4ST", "calls"] / br.loc["IPA4ST", "detections"], 1), 0.05)
        check("IPA11ST base rate 20.8% (next most extreme)", 20.8,
              round(100 * br.loc["IPA11ST", "calls"] / br.loc["IPA11ST", "detections"], 1), 0.05)
        check("IPA4ST is the lowest base rate of sixteen", "IPA4ST",
              str((a16["calls"] / a16["detections"]).idxmin() and
                  a16.loc[(a16["calls"] / a16["detections"]).idxmin(), "station"]))
    cu = maybe("data/outputs/v13_runs/colab_unfreeze/out_frozen.csv")
    cb = maybe("data/outputs/v13_runs/colab_unfreeze/out_block34.csv")
    if cu is not None and cb is not None:
        f4 = cu[cu["station"] == "IPA4ST"].iloc[0]
        b4 = cb[cb["station"] == "IPA4ST"].iloc[0]
        check("knife edge: 1,059 FPs kept -> 0.085", 0.085,
              round(float(f4["v13_precision"]), 3), 0.001)
        check("knife edge: 45 FPs kept -> 0.690", 0.690,
              round(float(b4["v13_precision"]), 3), 0.001)
        check("FPs kept, frozen (1,059)", 1059, int(f4["kept_false_positives"]))
        check("FPs kept, blocks 3+4 (45)", 45, int(b4["kept_false_positives"]))
        check("IPA4ST false positives (2,370)", 2370, int(f4["false_positives"]))

    # ---- the scan comparison written today ----
    s = maybe("data/outputs/scan_vs_heldout.csv")
    if s is not None:
        s = s.set_index(["site", "arm"])
        for site, armn, det, rec, mp in (
                ("IPA4ST", "ftfrozen", 472, 91, 0.875),
                ("IPA4ST", "ftfinetuned", 635, 91, 0.859),
                ("IPA4ST", "b34", 516, 91, 0.875),
                ("IPA20ST", "ftfrozen", 1029, 416, 0.959),
                ("IPA20ST", "ftfinetuned", 864, 416, 0.961),
                ("IPA20ST", "b34", 751, 405, 0.964),
                ("IPA13ST", "ftfrozen", 785, 184, None),
                ("IPA13ST", "ftfinetuned", 505, 184, None),
                ("IPA13ST", "b34", 510, 184, None)):
            if (site, armn) not in s.index:
                check(f"{site}/{armn} scan", det, None)
                continue
            r = s.loc[(site, armn)]
            check(f"{site}/{armn} detections", det, int(r["detections"]))
            check(f"{site}/{armn} calls recovered", rec, int(r["calls_recovered"]))
            if mp is not None:
                check(f"{site}/{armn} matched precision", mp,
                      round(float(r["matched_precision"]), 3), 0.001)

    u = maybe("data/outputs/scan_unique_windows.csv")
    if u is not None:
        q = u[(u["site"] == "IPA4ST") & (u["arm"] == "b34") & (u["vs"] == "ftfrozen")]
        check("b34-only windows at IPA4ST (103)", 103, int(q["unique_to_arm"].iloc[0]))
        check("of those never reviewed (97)", 97,
              int(q["of_those_never_reviewed"].iloc[0]))
        q = u[(u["site"] == "IPA4ST") & (u["arm"] == "ftfrozen") & (u["vs"] == "b34")]
        check("frozen-only windows at IPA4ST (74)", 74, int(q["unique_to_arm"].iloc[0]))
        check("of those never reviewed (66)", 66,
              int(q["of_those_never_reviewed"].iloc[0]))

    # ---- spatial non-independence, limitation 5 ----
    r = rev[rev["verdict"] == "call"].copy()
    r["start_s"] = r["start_s"].astype(float)
    r["ts"] = pd.to_datetime(r["timestamp"], errors="coerce")
    if r["ts"].notna().all():
        r["abs_s"] = r["ts"].astype("int64") // 10**9 + r["start_s"]
        for tol, claimed in ((2, 2.5), (5, 5.0), (300, 58.0)):
            hit = 0
            for site, sub in r.groupby("site"):
                other = r[r["site"] != site]["abs_s"].values
                for v in sub["abs_s"].values:
                    if ((other >= v - tol) & (other <= v + tol)).any():
                        hit += 1
            check(f"same call at another station within +/-{tol}s", claimed,
                  round(100 * hit / len(r), 1), 0.15)
    else:
        check("spatial non-independence", "2.5/5.0/58", None)

    # ---- the sixteen-fold head ablation ----
    #
    # Recomputed from the four arms rather than trusted, because these are the
    # numbers that settled a submission blocker and the abstract now quotes two
    # of them. The paired statistics are what the text reports; the means in
    # Table 3 are checked separately, cell by cell, below.
    import glob
    import numpy as np
    abl = {}
    for p in sorted(glob.glob(os.path.join(
            REPO, "data/outputs/v13_runs/ablations_2026-08-19/abl_*.csv"))):
        abl[os.path.basename(p)[4:-4]] = pd.read_csv(p).set_index("station")
    if len(abl) == 4:
        sts = sorted(abl["freqpos"].index)
        same = len({tuple(int(abl[k].loc[s, "gated_detections"]) for s in sts)
                    for k in abl}) == 1
        check("ablation arms share an evaluation set", True, same)
        check("ablation folds", 16, len(sts))

        def paired(x, y, m="gated_loso_precision"):
            d = np.array([abl[x].loc[s, m] - abl[y].loc[s, m] for s in sts])
            se = d.std(ddof=1) / np.sqrt(len(d))
            return d.mean(), d.mean() / se, int((d > 0).sum())

        for x, y, want_mean, want_t, want_win in [
                ("freq", "temporal", 0.0138, 2.63, 11),
                ("freqpos", "freq", -0.0012, -0.26, 5),
                ("freqpos", "temporal", 0.0126, 2.61, 12)]:
            m, t, w = paired(x, y)
            check(f"{x} - {y} precision", want_mean, round(m, 4), 0.0001)
            check(f"  its t", want_t, round(t, 2), 0.01)
            check(f"  folds won", want_win, w)
        m, t, w = paired("freqpos", "temporal", "gated_loso_calls_retained")
        check("freqpos - temporal recall", 0.0362, round(m, 4), 0.0001)
        check("  its t", 2.03, round(t, 2), 0.01)

        for arm, want in [("temporal", 0.9566), ("freq", 0.9703),
                          ("freqpos", 0.9692), ("freqpos_noconfuser", 0.9586)]:
            check(f"Table 3 precision, {arm}", want,
                  round(abl[arm]["gated_loso_precision"].mean(), 4), 0.0001)
        # the band-split figure is the one the abstract still quotes; the
        # position-encoding one was withdrawn on 2026-08-20 when the same two
        # arms on a later dataset reversed its sign, so the manuscript now
        # reports both nulls rather than either as a finding.
        for s in ("+0.0138", "+2.63"):
            check(f"abstract prints {s}", True, s in tex)
        for s in ("-0.0012", "-0.26", "-0.0030", "-0.90", "+0.0005", "+0.14"):
            check(f"body reports {s}", True, s in tex)
        check("no recommendation is made between the heads", True,
              "no recommendation between the two" in tex)
        check("the withdrawn advice is gone", True,
              "should use\n\\texttt{temporal\\_freq}" not in tex)

    # ---- the same two heads on the later dataset ----
    #
    # A second measurement of one comparison, and the reason the first was
    # withdrawn. The dataset differs -- 1,056 expert bird clips and 27 field
    # C. pogonias were added -- so this is not a replication; it is a second
    # internally clean comparison whose sign disagrees with the first.
    later = {}
    for a in ("freq", "freqpos"):
        p = os.path.join(REPO,
                         f"data/outputs/v13_runs/full_2026-08-19/"
                         f"loso16_{a}_evalfix.csv")
        if os.path.exists(p):
            later[a] = pd.read_csv(p).set_index("station")
    if len(later) == 2:
        sts2 = sorted(later["freq"].index)
        check("later run: arms share an evaluation set", True,
              all(later["freq"].loc[s, "gated_detections"]
                  == later["freqpos"].loc[s, "gated_detections"] for s in sts2))
        # Half a unit in the last printed place, against the UNROUNDED mean.
        # Until 2026-08-30 this compared round(mean, 4) with a constant that had
        # itself been produced by round(mean, 4), so it could not detect a
        # mis-rounded figure -- and it did not: the freq arm's exact mean is
        # 0.96334999..., whose correct rendering is 0.9633, while round() gave
        # 0.9634 and the manuscript printed that for a day.
        for a, want in (("freq", 0.9633), ("freqpos", 0.9604)):
            check(f"later run: {a} precision", want,
                  later[a]["gated_loso_precision"].mean(), 0.00005)
        d2 = np.array([later["freq"].loc[s, "gated_loso_precision"]
                       - later["freqpos"].loc[s, "gated_loso_precision"]
                       for s in sts2])
        se2 = d2.std(ddof=1) / np.sqrt(len(d2))
        check("later run: freq - freqpos", 0.0030, round(d2.mean(), 4), 0.0001)
        check("  its t", 0.90, round(d2.mean() / se2, 2), 0.01)
        # The same contrast against the frozen replicate, which is what
        # licenses the sentence saying the sign turns on run selection. If
        # these two ever came out with the same sign, that sentence is wrong.
        _rp2 = maybe("data/outputs/v13_runs/full_2026-08-19/"
                     "loso16_freqpos_replicate.csv")
        if _rp2 is not None:
            _rp2 = _rp2.set_index("station")
            d3 = np.array([later["freq"].loc[s, "gated_loso_precision"]
                           - _rp2.loc[s, "gated_loso_precision"] for s in sts2])
            check("later run: freq - freqpos replicate", -0.0005,
                  round(d3.mean(), 4), 0.0001)
            check("  its t", -0.14,
                  round(d3.mean() / (d3.std(ddof=1) / np.sqrt(len(d3))), 2), 0.01)
            check("  and the two pairings disagree in sign", True,
                  bool(d2.mean() * d3.mean() < 0))
            check("  both are smaller than the measured floor", True,
                  bool(max(abs(d2.mean()), abs(d3.mean())) < 0.0035))
        # Both differences must be taken the same way round. paired() above is
        # called as (freqpos, freq) elsewhere, which is the opposite subtraction
        # from d2, and comparing those two signs compares a quantity with its own
        # negation -- it agrees whatever the data does.
        first = paired("freq", "freqpos")[0]
        # Until 2026-08-29 this asserted the two builds DISAGREED, which was
        # true only while the later pair was scored on the retired evaluation
        # pool. On the corrected pool they agree, and the reversal that
        # survives is the one between two draws of a single specification.
        check("  the sign now AGREES with the first run", True,
              (d2.mean() < 0) == (first < 0),
              )
        # +0.0012 taken this way round. The manuscript prints -0.0012 because
        # it describes the encoding being *added* to the split, which is
        # freqpos - freq. Same number, opposite subtraction, and worth pinning
        # here because getting it backwards is how the sign check above passed
        # against its own negation.
        check("  first run freq - freqpos", 0.0012, round(first, 4), 0.0001)
    else:
        check("head ablation", "4 arms", f"{len(abl)} found")

    # ---- the gate ablation ----
    #
    # Recomputed from the two scans rather than trusted, because this is the
    # paper's only controlled measurement of the filter it describes, and
    # because the per-class split is the part that carries the claim: the
    # headline percentages range from 41 to 94 and mean almost nothing on their
    # own.
    DRV = os.path.join(REPO, "data/outputs/detection_review")
    gate = {}
    for st in ("IPA1ST", "IPA2ST", "IPA4ST"):
        g = os.path.join(DRV, f"{st}_0500-1900_full819_all_detections.csv")
        u = os.path.join(DRV, f"{st}_0500-1900_nogate_all_detections.csv")
        if os.path.exists(g) and os.path.exists(u):
            gate[st] = (pd.read_csv(g), pd.read_csv(u))
    if len(gate) == 3:
        tot_g = sum(len(a) for a, _ in gate.values())
        tot_u = sum(len(b) for _, b in gate.values())
        check("gate: gated total (674)", 674, tot_g)
        check("gate: ungated total (3,962)", 3962, tot_u)
        check("gate: removed overall (83.0%)", 83.0,
              round(100 * (tot_u - tot_g) / tot_u, 1), 0.05)
        for st, want in (("IPA1ST", 93.8), ("IPA2ST", 41.3), ("IPA4ST", 60.5)):
            a, b = gate[st]
            check(f"gate: {st} removed", want,
                  round(100 * (len(b) - len(a)) / len(b), 1), 0.05)
        # the four-fold split between the classes, which is the finding
        for st, sp, want in (("IPA1ST", "C_pogonias", 99.2),
                             ("IPA2ST", "C_pogonias", 91.3),
                             ("IPA4ST", "C_pogonias", 89.6),
                             ("IPA1ST", "Cernic", 26.3),
                             ("IPA2ST", "Cernic", 32.9),
                             ("IPA4ST", "Cernic", 25.1)):
            a, b = gate[st]
            na, nb = int((a.species == sp).sum()), int((b.species == sp).sum())
            check(f"gate: {st} {sp} removed", want,
                  round(100 * (nb - na) / nb, 1), 0.05)
        check("every pogonias rate exceeds every nictitans rate", True,
              min(100 * (int((b.species == "C_pogonias").sum())
                         - int((a.species == "C_pogonias").sum()))
                  / max(1, int((b.species == "C_pogonias").sum()))
                  for a, b in gate.values())
              > max(100 * (int((b.species == "Cernic").sum())
                           - int((a.species == "Cernic").sum()))
                    / max(1, int((b.species == "Cernic").sum()))
                    for a, b in gate.values()))
        # IPA4ST removed none of its four Colobus windows
        a4, b4 = gate["IPA4ST"]
        check("gate: IPA4ST removed no Colobus", 0,
              int((b4.species == "Colobus_guereza").sum())
              - int((a4.species == "Colobus_guereza").sum()))
        for s in ("93.8", "41.3", "60.5", "99.2", "89.6", "25.1", "1.61",
                  "3\\,288", "3\\,962"):
            check(f"manuscript prints {s}", True, s in tex)

    # ---- what the negative class is made of ----
    #
    # Recomputed from the packed index rather than trusted, because these are
    # the numbers that turn over fastest: every expert verdict moves a clip
    # between classes, and the paragraph describing the class is prose that no
    # build step regenerates.
    idx = maybe("data/outputs/v13_runs/full_2026-08-19/v13_index.csv")
    if idx is not None:
        ok = idx["ok"].astype(str).str.lower().isin(("true", "1"))
        bg = idx[ok & (idx["label"] == "Background")].drop_duplicates(
            subset="path")
        src = bg["source"].astype(str)
        ref = src.str.startswith("reference:")
        birds = src.str.startswith("expert_birds:")
        unheard = 155   # the auto_flagged_fp clips the expert's pass missed
        # Four categories now, not three. The 1,056 bird clips are counted
        # apart from the 7,003 because the expert chose them by species rather
        # than adjudicating each one, which is a weaker guarantee; folding them
        # into the listened-to count would overstate what is known about them.
        check("Background clips (10,165)", 10165, len(bg))
        check("  of them curated non-target (1,951)", 1951, int(ref.sum()))
        check("  of them expert-selected birds (1,056)", 1056, int(birds.sum()))
        check("  of them listened to (7,003)", 7003,
              int((~ref & ~birds).sum()) - unheard)
        # Was check(155, unheard) with unheard = 155 defined ten lines up:
        # a constant against itself. What can actually fail is whether the
        # manuscript still prints it and whether the four categories still
        # add up to the class.
        check("  manuscript prints 155 unheard", True, "155 are field" in tex)
        check("  the four categories account for every clip", len(bg),
              7003 + 1056 + 1951 + 155)
        check("no BirdNET left in the negative class", 0,
              int(bg["source"].str.contains("birdnet", case=False).sum()))
        for s in ("10{,}165", "7{,}003", "1{,}056", "1{,}951", "16{,}826"):
            check(f"manuscript prints {s}", True, s in tex)

        # ---- Table 1, as trained ----
        #
        # Until 2026-08-25 this table printed the previous build: 3,002 Cernic,
        # 665 guereza, 150 pogonias, 907 confuser, and a 25,891-clip Background
        # whose largest block was 16,826 machine-labelled clips the shipped
        # build dropped -- all under a "human-verified" caption. Nothing here
        # was watching it. Now every row is pinned to the shipped index.
        a0 = idx[ok & (idx["aug"] == 0)].drop_duplicates(subset="path")
        per = a0.groupby("label").size().to_dict()
        for lab, want in (("Cernic", 3004), ("Colobus_guereza", 1440),
                          ("C_pogonias", 177), ("Colobus_confuser", 961),
                          ("Background", 10165)):
            check(f"Table 1 {lab} originals ({want:,})", want,
                  int(per.get(lab, -1)))
        check("Table 1 total (15,747)", 15747, int(sum(per.values())))
        for s in ("3{,}004", "1{,}440", "15{,}747"):
            check(f"manuscript prints {s}", True, s in tex)
        # The two audited blocks. 2,370 is the size of an audit that happened:
        # every clip IPA4ST contributed as a reviewed false positive at the
        # time was listened to, and three were calls. It is not recomputed from
        # the current run, because renumbering a completed audit to match a
        # later denominator would assert that clips were heard which may not
        # have been. What is checked instead is that the manuscript still
        # prints the audited size, and, separately and visibly, how far the
        # station's reviewed-FP count has drifted from it since.
        #
        # Until 2026-08-25 the first line here read check(..., 2370, 2370) --
        # a constant against itself, which cannot fail and so measured nothing.
        audited = 2370
        check("IPA4ST negatives audited (2,370)", True,
              "2{,}370" in tex)
        check("  of which genuine calls (3, 0.13%)", 0.13,
              round(100 * 3 / audited, 2), 0.005)
        _loso = maybe("data/outputs/v13_runs/full_2026-08-19/loso16_freqpos_evalfix.csv")
        if _loso is not None:
            r4 = _loso[_loso["station"] == "IPA4ST"]
            if len(r4):
                now = int(r4["detections"].iloc[0] - r4["calls"].iloc[0])
                # Not a failure: a note that the audit no longer covers the
                # whole pool. If this grows, the sentence needs the words
                # "of the" rather than "All".
                # Reported, not asserted. This compares two different things
                # on purpose -- the size of a completed audit against the
                # station's current reviewed-FP count -- so a tolerance here
                # is a silent judgement about how much drift is acceptable.
                # It printed OK across 2,370 against 2,401 under an
                # undisclosed +/-40 until an audit on 2026-08-26 read it.
                check("  IPA4ST reviewed FPs now (audit covered 2,370)",
                      now, now)
                if now != audited:
                    print(f"    note: the audit covered {audited} clips and "
                          f"the station now contributes {now}. If that gap "
                          f"grows, the manuscript's \"All 2,370\" has to "
                          f"become \"2,370 of the\".")
        # Was 3143 against 3143. The audit file is on disk and has one row
        # per clip the expert heard, so count it.
        _audit = maybe("data/labels/auto_flagged_fp_review_2026-08-18.csv")
        check("auto_flagged_fp audited (3,143)", 3143,
              len(_audit) if _audit is not None else None)
        check("  of which not noise (6, 0.19%)", 0.19,
              round(100 * 6 / 3143, 2), 0.005)

    # ---- the sixteen-fold four-arm sweep (limitations, fourth item) ----
    #
    # The nopogonias arm exists twice on disk: the 2026-08-28 run under
    # unfreeze_2026-08-21_drive/nopogonias/ trained on 23 of its own
    # evaluation windows and is retired; only nopogonias_fixed_2026-08-29
    # may back a manuscript figure. The +0.0135 check below reads the leaky
    # files on purpose -- the manuscript quotes that number as the size of
    # the artefact, and the claim should fail if those files change.
    from math import sqrt as _sqrt
    import glob as _glob
    _fz = maybe("data/outputs/v13_runs/full_2026-08-19/loso16_freqpos_evalfix.csv")
    _arms = {
        "block4": maybe("data/outputs/v13_runs/unfreeze_2026-08-21_drive/block4_loso16.csv"),
        "block34": maybe("data/outputs/v13_runs/unfreeze_2026-08-21_drive/block34_loso16.csv"),
        "nopogonias": maybe("data/outputs/v13_runs/nopogonias_fixed_2026-08-29/loso16_nopogonias.csv"),
        "nocolobus": maybe("data/outputs/v13_runs/nocolobus_2026-08-29/loso16_nocolobus.csv"),
    }
    if _fz is None or any(v is None for v in _arms.values()):
        check("four-arm sweep (sixteen-fold) source files", "present", None)
    else:
        _fz = _fz.set_index("station").sort_index()

        def _paired(_a, _col):
            _d = (_a[_col] - _fz[_col]).to_numpy()
            return (round(_d.mean(), 4),
                    round(_d.mean() / (_d.std(ddof=1) / _sqrt(len(_d))), 2))

        check("frozen macro gated precision (0.9604)", 0.9604,
              round(_fz.gated_loso_precision.mean(), 4))
        check("frozen macro calls kept (92.0%)", 92.0,
              round(100 * _fz.gated_loso_calls_retained.mean(), 1))
        _claim = {  # macro prec, calls kept %, paired d_prec, t
            "block4": (0.9654, 93.6, 0.0050, 1.74),
            "block34": (0.9713, 92.2, 0.0109, 2.45),
            "nopogonias": (0.9705, 87.4, 0.0102, 1.92),
            "nocolobus": (0.9680, 90.7, 0.0076, 1.37),
        }
        for _nm, _a in _arms.items():
            _a = _a.set_index("station").sort_index()
            for _c in ("detections", "gated_detections"):
                check(f"{_nm} eval pool identical to frozen ({_c}, 16/16)",
                      16, int((_a[_c] == _fz[_c]).sum()))
            _p, _k, _dp, _tp = _claim[_nm]
            check(f"{_nm} macro gated precision", _p,
                  round(_a.gated_loso_precision.mean(), 4))
            check(f"{_nm} macro calls kept %", _k,
                  round(100 * _a.gated_loso_calls_retained.mean(), 1))
            _got_dp, _got_tp = _paired(_a, "gated_loso_precision")
            check(f"{_nm} paired precision delta vs frozen", _dp, _got_dp)
            check(f"{_nm} paired precision t", _tp, _got_tp, 0.005)
            _arms[_nm] = _a
        _dr, _tr = _paired(_arms["nopogonias"], "gated_loso_calls_retained")
        check("nopogonias paired calls-kept delta (-0.0462)", -0.0462, _dr)
        check("nopogonias paired calls-kept t (-2.83)", -2.83, _tr, 0.005)
        _leak_files = sorted(_glob.glob(os.path.join(
            REPO, "data/outputs/v13_runs/unfreeze_2026-08-21_drive/nopogonias/*.csv")))
        if len(_leak_files) == 16:
            _lk = pd.concat(map(pd.read_csv, _leak_files)).set_index("station").sort_index()
            _got_dp, _ = _paired(_lk, "gated_loso_precision")
            check("retired leaky nopogonias delta the tex quotes (+0.0135)",
                  0.0135, _got_dp)
        else:
            check("retired leaky nopogonias files (16 stations)", 16, None)
        for _s in ("0.9604", "0.9654", "0.9713", "0.9705", "0.9680", "93.6",
                   "92.2", "87.4", "90.7", "+0.0050", "+0.0109", "+0.0102",
                   "+0.0076", "t = +1.37", "-0.0462", "-0.0248",
                   "-0.0304", "t = -2.42", "+0.0105", "t = +2.54",
                   "nopogonias\\_fixed\\_2026-08-29"):
            check(f"sweep figure in tex: {_s}", 1, int(_s in tex))

    # ---- the measured run-to-run noise floor ----
    #
    # Two independent unseeded draws of one specification. This is the number
    # the whole four-arm conclusion is read against, so it is recomputed here
    # rather than quoted: if the replicate CSV changes, every "not separable
    # from noise" sentence in the limitations section has to be re-argued.
    _rep = maybe("data/outputs/v13_runs/full_2026-08-19/loso16_freqpos_replicate.csv")
    if _rep is None or _fz is None:
        check("noise-floor replicate present", "present", None)
    else:
        _rep = _rep.set_index("station").sort_index()
        for _c in ("detections", "gated_detections"):
            check(f"replicate scores the same pool as the arm it replicates ({_c})",
                  16, int((_rep[_c] == _fz[_c]).sum()))
        for _col, _dm, _dt, _sd, _mx in (
                ("gated_loso_precision", 0.0035, 1.34, 0.0103, 0.0256),
                ("gated_loso_calls_retained", -0.0039, -0.33, 0.046, 0.125)):
            _d = (_rep[_col] - _fz[_col]).to_numpy()
            _se = _d.std(ddof=1) / _sqrt(len(_d))
            check(f"noise floor {_col}: paired mean", _dm, round(_d.mean(), 4))
            check("  its t", _dt, round(_d.mean() / _se, 2), 0.005)
            check("  its per-station SD", _sd, round(_d.std(ddof=1), 4), 0.0005)
            check("  its largest single-station move", _mx,
                  round(abs(_d).max(), 4), 0.0005)
        # The floor must stay below the effects it is used to judge, or the
        # sentences that judge them are wrong in the other direction.
        _floor = abs((_rep.gated_loso_precision - _fz.gated_loso_precision).mean())
        # The Colobus-free arm's t is quoted in the text as matching the
        # floor's to two decimals. If either moves, that sentence is wrong.
        _dk, _tk = _paired(_arms["nocolobus"], "gated_loso_calls_retained")
        check("nocolobus paired calls-kept delta (-0.0133)", -0.0133, _dk)
        check("nocolobus paired calls-kept t (-1.31)", -1.31, _tk, 0.005)
        for _nm, _eff in (("block4", 0.0050), ("block34", 0.0109),
                          ("nopogonias", 0.0102), ("nocolobus", 0.0076)):
            check(f"{_nm} effect as a multiple of the floor", True,
                  bool(_eff > _floor))
        for _s in ("+0.0035", "0.0103", "0.0256", "0.0039", "0.0115",
                   "0.0076", "0.125", r"loso16\_freqpos\_replicate.csv"):
            check(f"noise floor figure in tex: {_s}", 1, int(_s in tex))

    # ---- the Colobus OOD positive control ----
    #
    # This block exists because its absence let a false sentence stand for
    # nineteen days. The manuscript names one fitted parameter a reproducer
    # must copy -- the 97th-percentile override for Colobus_guereza -- and it
    # justified that parameter by the behaviour of a statistics file that was
    # overwritten under its own filename on 2026-08-20. Nothing recomputed the
    # controls afterwards, so nothing noticed. Now the claim is a number.
    _ctl = maybe("data/outputs/colobus_ood_controls.csv")
    if _ctl is None:
        check("Colobus OOD control scores", "present", None)
    else:
        _best = _ctl[_ctl["convention"] == "best"].set_index("stats")
        check("control clips scored (9)", 9, int(_ctl["n_controls"].iloc[0]))
        check("shipped statistics files scored (5)", 5, len(_best))
        # The two cutoffs the manuscript prints for the head the override was
        # fitted for.
        check("fold_IPA4ST Colobus p90 (283.7)", 283.7,
              float(_best.loc["fold_IPA4ST", "p90"]), 0.05)
        check("fold_IPA4ST Colobus p97 (377.8)", 377.8,
              float(_best.loc["fold_IPA4ST", "p97"]), 0.05)
        # "one of the nine roars passes, at either percentile" -- the sentence
        # that replaced "all nine pass".
        for _q in ("pass_p90", "pass_p97"):
            check(f"fold_IPA4ST controls admitted, {_q} (1 of 9)", 1,
                  int(_best.loc["fold_IPA4ST", _q]))
        # "the best any percentile achieves is five of nine"
        check("best across all shipped heads at p90 or p97 (5 of 9)", 5,
              int(max(_best["pass_p90"].max(), _best["pass_p97"].max())))
        # "only the 99th reaches nine on any single head"
        check("heads where p99 admits all nine (1)", 1,
              int((_best["pass_p99"] == 9).sum()))
        check("heads where p97 admits all nine (0)", 0,
              int((_best["pass_p97"] == 9).sum()))
        # The retired statistics the justification was fitted on must not
        # reappear: if some file ever ships with these values again, the
        # retraction has to be revisited rather than silently contradicted.
        check("no shipped head still carries the retired p90 202.9", 0,
              int((abs(_best["p90"] - 202.9) < 0.1).sum()))
        for _s in ("283.7", "377.8", "202.9", "328.4",
                   r"score\_colobus\_controls.py"):
            check(f"OOD retraction figure in tex: {_s}", 1, int(_s in tex))

    # ---- the head ablation, threshold-free and at fitted thresholds ----
    #
    # This block exists because three independent readers found the first
    # write-up of this comparison overclaiming, and every one of their
    # objections was a number nobody had computed. The paired figures at fitted
    # thresholds are not wrong, they are answers to a different question: the
    # two arms are scored at mean thresholds 0.287 and 0.544, so part of the
    # difference is the operating point. Both are recomputed here.
    _pr = maybe("data/outputs/v13_runs/full_2026-08-19/head_ablation_prauc.csv")
    _tmp = maybe("data/outputs/v13_runs/full_2026-08-19/loso16_temporal_evalfix.csv")
    if _pr is None or _tmp is None or _fz is None:
        check("head ablation, corrected pool", "present", None)
    else:
        _tmp = _tmp.set_index("station").sort_index()
        _frq = maybe("data/outputs/v13_runs/full_2026-08-19/loso16_freq_evalfix.csv")
        _frq = _frq.set_index("station").sort_index()
        check("temporal arm eval pool identical to frozen (16/16)", 16,
              int((_tmp["gated_detections"] == _fz["gated_detections"]).sum()))
        check("temporal macro gated precision (0.9537)", 0.9537,
              round(_tmp.gated_loso_precision.mean(), 4))
        # at fitted thresholds
        _dp = (_frq.gated_loso_precision - _tmp.gated_loso_precision)
        _dr = (_frq.gated_loso_calls_retained - _tmp.gated_loso_calls_retained)
        _se = _dp.std(ddof=1) / _sqrt(16)
        check("band split at fitted thresholds (+0.0096)", 0.0096,
              round(_dp.mean(), 4))
        check("  its t (+1.66)", 1.66, round(_dp.mean() / _se, 2), 0.005)
        check("  folds won (10)", 10, int((_dp > 0).sum()))
        check("mean fitted threshold, temporal (0.287)", 0.287,
              round(_tmp.gated_loso_threshold.mean(), 3), 0.0005)
        check("mean fitted threshold, freq (0.544)", 0.544,
              round(_frq.gated_loso_threshold.mean(), 3), 0.0005)
        check("precision/recall anti-correlation (-0.59)", -0.59,
              round(_dp.corr(_dr), 2), 0.005)
        check("stations gaining on both metrics (3)", 3,
              int(((_dp > 0) & (_dr > 0)).sum()))
        _o = _dp.sort_values(ascending=False)
        check("mean without the largest station (+0.0060)", 0.0060,
              round(_dp.drop(_o.index[0]).mean(), 4))
        check("mean without the largest two (+0.0037)", 0.0037,
              round(_dp.drop(list(_o.index[:2])).mean(), 4))
        check("outside each metric's top three, precision (+0.0013)", 0.0013,
              round(_dp.drop(list(_dp.sort_values(ascending=False).index[:3])).mean(), 4))
        check("outside each metric's top three, recall (+0.0075)", 0.0075,
              round(_dr.drop(list(_dr.sort_values(ascending=False).index[:3])).mean(), 4))
        # threshold-free -- the measurement the passage now leads with
        _piv = _pr.pivot(index="station", columns="arm", values="ap")
        for _a, _want in (("temporal", 0.9787), ("freq", 0.9808), ("freqpos", 0.9825)):
            check(f"macro average precision, {_a}", _want,
                  round(_piv[_a].mean(), 4))
        _d2 = (_piv["freq"] - _piv["temporal"])
        _se2 = _d2.std(ddof=1) / _sqrt(len(_d2))
        check("band split THRESHOLD-FREE (+0.0021)", 0.0021, round(_d2.mean(), 4))
        check("  its t (+1.36)", 1.36, round(_d2.mean() / _se2, 2), 0.005)
        check("  folds won (12)", 12, int((_d2 > 0).sum()))
        check("  the fitted-threshold figure is 4.6x larger", 4.6,
              round(_dp.mean() / _d2.mean(), 1), 0.05)
        _d3 = (_piv["freqpos"] - _piv["freq"])
        _se3 = _d3.std(ddof=1) / _sqrt(len(_d3))
        check("CoordConv threshold-free (+0.0017)", 0.0017, round(_d3.mean(), 4))
        check("  its t (+0.70)", 0.70, round(_d3.mean() / _se3, 2), 0.005)
        check("  folds won (7)", 7, int((_d3 > 0).sum()))
        # cross-build agreement, the reason the ablation build is called
        # superseded rather than corroborating
        _ab = {}
        for _n in ("temporal", "freq"):
            _p = maybe(f"data/outputs/v13_runs/ablations_2026-08-19/abl_{_n}.csv")
            if _p is not None:
                _ab[_n] = _p.set_index("station").sort_index()
        if len(_ab) == 2:
            _d0 = (_ab["freq"].gated_loso_precision
                   - _ab["temporal"].gated_loso_precision)
            check("cross-build per-station correlation (0.38)", 0.38,
                  round(_d0.corr(_dp), 2), 0.005)
            _nz = (_d0 != 0) & (_dp != 0)
            check("  stations moving in both builds (14)", 14, int(_nz.sum()))
            check("  of which agree in sign (9)", 9,
                  int((np.sign(_d0[_nz]) == np.sign(_dp[_nz])).sum()))
            check("ablation build gated pool (3,476)", 3476,
                  int(_ab["temporal"].gated_detections.sum()))
        # Line-wrap-insensitive: the manuscript wraps at 79 columns, so a
        # phrase check that assumes a single space fails on a break. Collapse
        # whitespace in the haystack once and search that.
        _flat = " ".join(tex.split())
        for _s in ("+0.0021", "t = +1.36", "twelve of sixteen", "0.287",
                   "0.544", "-0.59", "4.6 times", r"3\,476",
                   r"prauc\_head\_ablation.py"):
            check(f"head ablation figure in tex: {_s}", 1, int(_s in _flat))

    # ---- Table 3's second panel, cell by cell ----
    #
    # The table carried only the ablation build until 2026-08-30, under a
    # caption promising "identical data and evaluation sets" while the body
    # reported the same three heads on a different build. Both panels now print,
    # and both are checked against half a unit in the last printed place of the
    # unrounded mean -- not against round(), which cannot catch a mis-rounded
    # figure because it produces one. That is how 0.9634 stood for a day where
    # the exact mean is 0.96334999.
    _panel = {
        "temporal": ("full_2026-08-19/loso16_temporal_evalfix.csv", 0.9537, 0.8839),
        "temporal_freq": ("full_2026-08-19/loso16_freq_evalfix.csv", 0.9633, 0.9146),
        "temporal_freqpos": ("full_2026-08-19/loso16_freqpos_evalfix.csv", 0.9604, 0.9202),
        "second draw": ("full_2026-08-19/loso16_freqpos_replicate.csv", 0.9638, 0.9163),
    }
    for _nm, (_p, _wp, _wr) in _panel.items():
        _d = maybe("data/outputs/v13_runs/" + _p)
        if _d is None:
            check(f"Table 3 lower panel, {_nm}", "present", None)
            continue
        check(f"Table 3 lower panel, {_nm} precision", _wp,
              _d.gated_loso_precision.mean(), 0.00005)
        check(f"Table 3 lower panel, {_nm} recall", _wr,
              _d.gated_loso_calls_retained.mean(), 0.00005)
        check(f"  and {_wp} is printed in the tex", 1, int(str(_wp) in tex))
    # The caption's promise is now panel-local, so the two pool sizes must both
    # appear or the warning not to compare across panels loses its evidence.
    for _s in (r"3\,476", r"3\,497"):
        check(f"Table 3 caption names pool size {_s}", 1, int(_s in tex))
    # And the retired figures must not have crept back into the header comment.
    check("header comment no longer asserts the superseded +0.0138", 1,
          int("SUPERSEDED 2026-08-30" in tex))

    # ---- the BirdNET non-baseline, stated in the Background ----
    #
    # The one claim in the paper that answers "why build this at all". It is
    # recomputed rather than quoted because it is load-bearing for the
    # motivation and nothing else in this file touches the 2026-08-17 build.
    _wb = maybe("data/outputs/v13_runs/withbirdnet_2026-08-17/v13_index.csv")
    if _wb is None:
        check("BirdNET labels on this array", "present", None)
    else:
        _b = _wb[_wb["source"].astype(str).str.startswith("birdnet")]
        _lab = _b["source"].astype(str).str.split(":").str[-1]
        check("BirdNET detections on this array (17,101)", 17101, len(_b))
        check("  distinct labels (250)", 250, int(_lab.nunique()))
        check("  of which eBird species codes (249)", 249,
              int(_lab[_lab.str.fullmatch(r"[a-z]{4,8}[0-9]?")].nunique()))
        _prim = _lab[_lab.str.contains(
            "cercopith|colobus|monkey|guenon|primate|nictitans|pogonias|guereza",
            case=False, regex=True)]
        check("  of which primates (none)", 0, int(_prim.nunique()))
        for _s in (r"17\,101", "250 distinct labels", "249 are eBird"):
            check(f"Background prints {_s}", 1, int(_s in " ".join(tex.split())))

    # ---- the floor on three draws, and pogonias on two ----
    #
    # Both arms finished 2026-08-31. Three draws of one specification give three
    # pairwise differences that must sum to zero, so the number that bounds
    # anything is the largest of them, not whichever one got quoted first. And
    # the pogonias arm is the only one run twice: its two draws disagree on the
    # size of the recall cost by more than the frozen draws disagree with each
    # other, which is why the manuscript now reports the average with an
    # interval rather than either draw alone.
    _d3 = maybe("data/outputs/v13_runs/replicates_2026-08-30/frozen_rep3/loso16.csv")
    _n2 = maybe("data/outputs/v13_runs/replicates_2026-08-30/nopogonias_rep2/loso16.csv")
    _n1 = maybe("data/outputs/v13_runs/nopogonias_fixed_2026-08-29/loso16_nopogonias.csv")
    if any(x is None for x in (_d3, _n2, _n1)) or _fz is None or _rep is None:
        check("third frozen draw and second pogonias draw", "present", None)
    else:
        _d3 = _d3.set_index("station").sort_index()
        _n2 = _n2.set_index("station").sort_index()
        _n1 = _n1.set_index("station").sort_index()
        for _nm, _d in (("frozen_rep3", _d3), ("nopogonias_rep2", _n2)):
            check(f"{_nm}: sixteen folds", 16, len(_d))
            check(f"{_nm}: eval pool identical to frozen", 16,
                  int((_d["detections"] == _fz["detections"]).sum()))
        _draws = {"d1": _fz, "d2": _rep, "d3": _d3}
        _pairs = (("d2", "d1"), ("d3", "d1"), ("d3", "d2"))
        for _c, _want in (("gated_loso_precision", [0.0, 0.0035, 0.0035]),
                          ("gated_loso_calls_retained", [0.0039, 0.0076, 0.0115])):
            # float(), not just round(): a numpy scalar stringifies as
            # "np.float64(0.0035)" and the comparison below is textual, so
            # without the cast this check fails on a repr rather than a value.
            _got = sorted(float(round(abs((_draws[a][_c] - _draws[b][_c]).mean()), 4))
                          for a, b in _pairs)
            check(f"floor pairwise |d|, {_c}", str(_want), str(_got))
        check("largest floor difference in precision (0.0035)", 0.0035,
              max(abs((_draws[a].gated_loso_precision
                       - _draws[b].gated_loso_precision).mean())
                  for a, b in _pairs), 0.00005)
        for _nm, _d, _wk, _wt in (("draw 1", _n1, -0.0462, -2.83),
                                  ("draw 2", _n2, -0.0248, -1.90)):
            _x = (_d.gated_loso_calls_retained
                  - _fz.gated_loso_calls_retained).to_numpy()
            _se = _x.std(ddof=1) / _sqrt(16)
            check(f"pogonias {_nm} calls-kept delta", _wk, _x.mean(), 0.00005)
            check(f"  its t", _wt, round(_x.mean() / _se, 2), 0.005)
        _cols = ["gated_loso_precision", "gated_loso_calls_retained"]
        _npm = (_n1[_cols] + _n2[_cols]) / 2
        _frm = (_fz[_cols] + _rep[_cols] + _d3[_cols]) / 3
        for _c, _wm, _wt in (("gated_loso_precision", 0.0105, 2.54),
                             ("gated_loso_calls_retained", -0.0304, -2.42)):
            _x = (_npm[_c] - _frm[_c]).to_numpy()
            _se = _x.std(ddof=1) / _sqrt(16)
            check(f"pogonias averaged, {_c}", _wm, _x.mean(), 0.00005)
            check(f"  its t", _wt, round(_x.mean() / _se, 2), 0.005)
            check("  and its 95% CI excludes zero", True,
                  bool(abs(_x.mean() / _se) > 2.131))

    # ---- the fine-tuned arms, threshold-free ----
    #
    # Added after the head ablation showed a fitted-threshold comparison can be
    # mostly threshold placement (the band split lost four fifths of its effect
    # that way). These two arms carry the sweep's only nominally significant
    # backbone effect and had never been checked for the same confound. Scored
    # from the weights the 2026-08-21 run saved, so no retraining was involved.
    _ap = maybe("data/outputs/v13_runs/unfreeze_2026-08-21_drive/arms_prauc.csv")
    _fzap = maybe("data/outputs/v13_runs/full_2026-08-19/head_ablation_prauc.csv")
    if _ap is None or _fzap is None or _fz is None:
        check("fine-tuned arms scored threshold-free", "present", None)
    else:
        _fzap = _fzap[_fzap["arm"] == "freqpos"].set_index("station")
        _want = {"block34": (16, 0.0073, 1.45, 12, 0.0109),
                 "block4": (15, 0.0048, 1.55, 11, 0.0050)}
        for _arm, _g in _ap.groupby("arm"):
            _g = _g.set_index("station")
            _n, _wm, _wt, _ww, _fitted = _want[_arm]
            check(f"{_arm} threshold-free: folds scored", _n, len(_g))
            # the evaluation pool has to be the frozen arm's, or the paired
            # comparison is between different questions
            check(f"  its pool matches frozen (gated detections)", len(_g),
                  int((_g["n"] == _fz.loc[_g.index, "gated_detections"]).sum()))
            check(f"  and the same call counts", len(_g),
                  int((_g["n_calls"] == _fzap.loc[_g.index, "n_calls"]).sum()))
            _x = np.array([_g.ap[_s] - _fzap.ap[_s] for _s in _g.index])
            _se = _x.std(ddof=1) / _sqrt(len(_x))
            check(f"  paired average-precision delta", _wm, _x.mean(), 0.00005)
            check(f"  its t", _wt, round(_x.mean() / _se, 2), 0.005)
            check(f"  folds won", _ww, int((_x > 0).sum()))
            # the sentence's claim: neither interval excludes zero
            check(f"  its 95% CI contains zero", True,
                  bool(abs(_x.mean() / _se) < 2.145))
            # and the claim that these are NOT the band split's 4.6x case
            check(f"  fitted-threshold figure is under 2x the threshold-free",
                  True, bool(_fitted / _x.mean() < 2.0))
        for _s in ("+0.0073", "t = +1.45", "+0.0048", "t = +1.55",
                   r"prauc\_arms.py"):
            check(f"threshold-free arm figure in tex: {_s}", 1,
                  int(_s in " ".join(tex.split())))

    # ---- report ----
    w = max(len(n) for _, n, _, _ in RESULTS)
    print()
    for status, name, claimed, actual in RESULTS:
        print(f"  {status:4s} {name:{w}s}  claimed {claimed}   actual {actual}")
    n_ok = sum(1 for s, *_ in RESULTS if s == "OK")
    n_off = sum(1 for s, *_ in RESULTS if s == "OFF")
    n_skip = sum(1 for s, *_ in RESULTS if s == "SKIP")
    print(f"\n{n_ok} OK, {n_off} OFF, {n_skip} SKIP  of {len(RESULTS)} checks")
    if n_off:
        print("\nOFF means the manuscript and the file disagree. Fix one of them.")
    # the tex is read only to confirm the file we are checking is the live one
    print(f"\nchecked against {os.path.relpath(TEX, REPO)}, "
          f"{len(tex.splitlines())} lines")
    # Exit nonzero on any disagreement. Until 2026-08-28 this script always
    # exited 0 -- a run with failing checks reported success to anything that
    # reads exit codes, which is every wrapper in this repo. That is the exact
    # silent-pass failure mode the docstring above was written against, found
    # by an audit that perturbed a Table 2 cell and watched the exit code stay
    # green. SKIPs stay 0: "not verified here" is not a failure, but it is
    # counted and printed so it cannot pass for coverage.
    return 1 if n_off else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
