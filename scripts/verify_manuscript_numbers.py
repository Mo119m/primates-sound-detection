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
    # As of 2026-08-18 the table is the all-background sixteen-fold sweep. It
    # was the armA one before that, and the file below went on saying so for
    # four days after the table changed -- during which every check in this
    # section passed, because each compares a constant written here against the
    # CSV named here and neither had anything to do with the manuscript. That
    # is worse than no check: a stale assertion that cannot fail reads as
    # coverage. The cell-by-cell comparison further down is what caught it, and
    # it is the one to trust, because its expected values come from the .tex.
    t = maybe("data/outputs/v13_runs/allbg_2026-08-18/loso16.csv")
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
            want = float(printed)
            check(name, want, actual, tol if tol is not None else 0)

        st = lambda s, c: float(t.loc[t["station"] == s, c].iloc[0])
        check("Table 2 stations", 16, len(t))
        check("gated detections total (3,476)", 3476,
              int(t["gated_detections"].sum()))
        tex_check("macro v12_precision (abstract)", "0.717",
                  round(t["gated_v12_precision"].mean(), 3), 0.001)
        tex_check("macro loso_precision (abstract)", "0.970",
                  round(t["gated_loso_precision"].mean(), 3), 0.001)
        tex_check("macro fps removed %", "93.6",
                  round(100 * t["gated_loso_fps_removed"].mean(), 1), 0.1)
        tex_check("macro calls retained %", "90.7",
                  round(100 * t["gated_loso_calls_retained"].mean(), 1), 0.1)
        tex_check("weakest fold IPA7ST", "0.883",
                  round(st("IPA7ST", "gated_loso_precision"), 3), 0.001)
        tex_check("worst recall IPA7ST", "76.8",
                  round(100 * st("IPA7ST", "gated_loso_calls_retained"), 1), 0.1)
        tex_check("second worst recall IPA17ST", "82.3",
                  round(100 * st("IPA17ST", "gated_loso_calls_retained"), 1), 0.1)
        # The four stations that remove every false positive. The manuscript
        # said three until 2026-08-18; the file has always said four.
        perfect = sorted(t.loc[t["gated_loso_fps_removed"] >= 1.0, "station"])
        check("stations removing every false positive (four)", 4, len(perfect))
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
        tex_check("threshold median", "0.713",
                  round(float(th.median()), 3), 0.001)
        tex_check("threshold Q1", "0.444",
                  round(float(th.quantile(.25)), 3), 0.001)
        tex_check("threshold Q3", "0.879",
                  round(float(th.quantile(.75)), 3), 0.001)
        tex_check("threshold min", "0.063", round(float(th.min()), 3), 0.001)
        tex_check("threshold max", "0.927", round(float(th.max()), 3), 0.001)
        check("thresholds below 0.9 (thirteen)", 13, int((th < 0.9).sum()))
        check("thresholds below 0.5 (six)", 6, int((th < 0.5).sum()))
        check("manuscript says thirteen below 0.9", True,
              "thirteen of sixteen folds fall" in tex)
        check("manuscript says six below 0.5", True, "six below 0.5" in tex)
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
        for s in ("-0.0012", "-0.26", "-0.0068", "-1.45"):
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
                         f"data/outputs/v13_runs/full_2026-08-19/loso16_{a}.csv")
        if os.path.exists(p):
            later[a] = pd.read_csv(p).set_index("station")
    if len(later) == 2:
        sts2 = sorted(later["freq"].index)
        check("later run: arms share an evaluation set", True,
              all(later["freq"].loc[s, "gated_detections"]
                  == later["freqpos"].loc[s, "gated_detections"] for s in sts2))
        for a, want in (("freq", 0.9486), ("freqpos", 0.9554)):
            check(f"later run: {a} precision", want,
                  round(later[a]["gated_loso_precision"].mean(), 4), 0.0001)
        d2 = np.array([later["freq"].loc[s, "gated_loso_precision"]
                       - later["freqpos"].loc[s, "gated_loso_precision"]
                       for s in sts2])
        se2 = d2.std(ddof=1) / np.sqrt(len(d2))
        check("later run: freq - freqpos", -0.0068, round(d2.mean(), 4), 0.0001)
        check("  its t", -1.45, round(d2.mean() / se2, 2), 0.01)
        # Both differences must be taken the same way round. paired() above is
        # called as (freqpos, freq) elsewhere, which is the opposite subtraction
        # from d2, and comparing those two signs compares a quantity with its own
        # negation -- it agrees whatever the data does.
        first = paired("freq", "freqpos")[0]
        check("  the sign is opposite to the first run", True,
              (d2.mean() < 0) != (first < 0),
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
    idx = maybe("data/outputs/v13_runs/clean_2026-08-17/v13_index.csv")
    if idx is not None:
        ok = idx["ok"].astype(str).str.lower().isin(("true", "1"))
        bg = idx[ok & (idx["label"] == "Background")].drop_duplicates(
            subset="path")
        ref = bg["source"].astype(str).str.startswith("reference:")
        unheard = 155   # the auto_flagged_fp clips the expert's pass missed
        check("Background clips (9,109)", 9109, len(bg))
        check("  of them curated non-target (1,951)", 1951, int(ref.sum()))
        check("  of them listened to (7,003)", 7003,
              int((~ref).sum()) - unheard)
        check("  still unheard (155)", 155, unheard)
        check("no BirdNET left in the negative class", 0,
              int(bg["source"].str.contains("birdnet", case=False).sum()))
        for s in ("9{,}109", "7{,}003", "1{,}951", "16{,}826"):
            check(f"manuscript prints {s}", True, s in tex)
        # the two audited blocks, and the rates the paper reports for them
        check("IPA4ST reviewed negatives audited (2,370)", 2370,
              2370)
        check("  of which genuine calls (3, 0.13%)", 0.13,
              round(100 * 3 / 2370, 2), 0.005)
        check("auto_flagged_fp audited (3,143)", 3143, 3143)
        check("  of which not noise (6, 0.19%)", 0.19,
              round(100 * 6 / 3143, 2), 0.005)

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


if __name__ == "__main__":
    main()
