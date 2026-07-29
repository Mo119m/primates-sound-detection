"""
Do the temporal-structure signals improve the review ordering?

Every earlier attempt to improve the cleanup was a threshold, and every one of
them died in leave-one-station-out evaluation because the signals are not on a
common scale across stations. The ranking is different: a signal enters as a
percentile rank within its own station and the ranks are averaged, so there is
no cutoff and no weight that could fail to carry to a new site. A candidate
signal can therefore only help the ordering or not help it -- which makes
testing one cheap, and makes this script worth running before believing any of
them.

Two honest-accounting rules are built in, because this is a search over signals
and a search fits the data it searches:

1. Direction. Whether a large value of a new signal means "more likely genuine"
   is often unclear in advance, so both directions are reported. Picking the
   better one is one fitted bit per signal, and ``--holdout`` checks it: the
   direction is chosen on the other stations and scored on the held-out one.
2. Selection. Choosing which signals to add by their score on all stations is
   the same in-sample mistake in a different place. The forward selection at the
   end is therefore also run leave-one-station-out, and it is that number --
   not the in-sample one -- that says whether a signal is worth reporting.

Usage:
    # cheapest: reuse the table the evaluation already wrote
    python scripts/rank_signals_experiment.py \\
        --matched data/outputs/auto_cleanup/cleanup_vs_review.csv

    # or rebuild it from the review and the cleanup output
    python scripts/rank_signals_experiment.py \\
        --review reviews/ --cleanup data/outputs/auto_cleanup

    # the fifteen stations excluding the one an untrained species overran
    python scripts/rank_signals_experiment.py --matched ... \\
        --exclude-station IPA4ST
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd  # noqa: E402
import cleanup_eval  # noqa: E402
import episode_features  # noqa: E402


def _site_col(df):
    return "site" if "site" in df.columns else "station"


def _mean_ap(df, signals, site_col):
    """Mean average precision across stations for one signal set."""
    per = cleanup_eval.effort_curve_by_station(df, signals=signals,
                                              site_col=site_col)
    if not len(per):
        return None, None, 0
    ap = pd.to_numeric(per["avg_precision"], errors="coerce")
    rand = pd.to_numeric(per["ap_if_random"], errors="coerce")
    return (round(float(ap.mean()), 4), round(float((ap - rand).mean()), 4),
            int((ap > rand).sum()))


def candidates_alone(df, cands, site_col):
    """Each candidate signal on its own, in both directions."""
    rows = {}
    for col in cands:
        for higher in (True, False):
            ap, adv, above = _mean_ap(df, {col: higher}, site_col)
            if ap is None:
                continue
            arrow = "high=genuine" if higher else "low=genuine"
            rows[f"{col} ({arrow})"] = {
                "mean_avg_precision": ap, "vs_arbitrary": adv,
                "stations_above_arbitrary": above,
            }
    return pd.DataFrame(rows).T.sort_values("mean_avg_precision", ascending=False)


def added_to_baseline(df, cands, site_col, base):
    """The four reported signals plus one candidate, against the four alone."""
    base_ap, _, _ = _mean_ap(df, base, site_col)
    rows = {"(the four reported signals)": {
        "mean_avg_precision": base_ap, "delta": 0.0, "stations_above_arbitrary": None}}
    for col in cands:
        for higher in (True, False):
            ap, _, above = _mean_ap(df, {**base, col: higher}, site_col)
            if ap is None:
                continue
            arrow = "high=genuine" if higher else "low=genuine"
            rows[f"+ {col} ({arrow})"] = {
                "mean_avg_precision": ap, "delta": round(ap - base_ap, 4),
                "stations_above_arbitrary": above,
            }
    return pd.DataFrame(rows).T.sort_values("mean_avg_precision", ascending=False)


def holdout_selection(df, cands, site_col, base, max_added=2):
    """
    Forward selection, leave-one-station-out.

    For each station: greedily add candidate signals using only the *other*
    stations, then score the resulting set on the held-out station. Compares
    against the four reported signals scored on the same held-out station, so
    the two columns are directly comparable. A signal set that only helps
    in-sample shows up here as a delta at or below zero.
    """
    stations = sorted(df[site_col].dropna().astype(str).unique())
    rows = []
    for st in stations:
        tune = df[df[site_col].astype(str) != st]
        held = df[df[site_col].astype(str) == st]
        if not len(tune) or not len(held):
            continue

        chosen = dict(base)
        for _ in range(max_added):
            cur, _, _ = _mean_ap(tune, chosen, site_col)
            best, best_key = cur, None
            for col in cands:
                if col in chosen:
                    continue
                for higher in (True, False):
                    ap, _, _ = _mean_ap(tune, {**chosen, col: higher}, site_col)
                    if ap is not None and (best is None or ap > best):
                        best, best_key = ap, (col, higher)
            if best_key is None:
                break
            chosen[best_key[0]] = best_key[1]

        ap_new, _, _ = _mean_ap(held, chosen, site_col)
        ap_base, _, _ = _mean_ap(held, base, site_col)
        added = [c for c in chosen if c not in base]
        rows.append({
            site_col: st,
            "detections": len(held),
            "added": ", ".join(f"{c}({'hi' if chosen[c] else 'lo'})"
                               for c in added) or "(nothing)",
            "ap_baseline": ap_base,
            "ap_with_added": ap_new,
            "delta": (round(ap_new - ap_base, 4)
                      if ap_new is not None and ap_base is not None else None),
        })
    out = pd.DataFrame(rows)
    if len(out):
        d = pd.to_numeric(out["delta"], errors="coerce")
        out.loc[len(out)] = {site_col: "MEAN (held out)", "detections": len(df),
                             "added": f"{int((d > 0).sum())}/{int(d.notna().sum())} "
                                      f"stations improved",
                             "ap_baseline": round(float(pd.to_numeric(
                                 out["ap_baseline"], errors="coerce").mean()), 4),
                             "ap_with_added": round(float(pd.to_numeric(
                                 out["ap_with_added"], errors="coerce").mean()), 4),
                             "delta": round(float(d.mean()), 4)}
    return out


def main():
    ap_ = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--matched", default=None,
                     help="cleanup_vs_review.csv from a previous evaluation run "
                          "(the cheapest input; no model or audio needed).")
    ap_.add_argument("--review", default=None, help="Folder of review CSVs.")
    ap_.add_argument("--cleanup", default=None, help="Cleanup output folder.")
    ap_.add_argument("--exclude-station", default=None,
                     help="Comma-separated stations to drop before scoring.")
    ap_.add_argument("--gap-s", type=float,
                     default=episode_features.DEFAULT_GAP_S,
                     help="Episode gap in seconds (default: 300).")
    ap_.add_argument("--max-added", type=int, default=2,
                     help="How many candidates forward selection may add.")
    ap_.add_argument("--out", default=None, help="Folder for the CSVs.")
    args = ap_.parse_args()

    if args.matched:
        matched = pd.read_csv(args.matched)
    elif args.review and args.cleanup:
        matched, _ = cleanup_eval.run(args.review, args.cleanup)
    else:
        ap_.error("give either --matched or both --review and --cleanup")

    site_col = _site_col(matched)
    excluded = [s.strip() for s in (args.exclude_station or "").split(",")
                if s.strip()]
    if excluded:
        before = len(matched)
        matched = matched[~matched[site_col].astype(str).isin(excluded)].copy()
        print(f"Excluding {', '.join(excluded)}: {before} -> {len(matched)} "
              f"detections across {matched[site_col].nunique()} stations.\n")

    if "cleanup" not in matched.columns:
        matched["cleanup"] = "clean"

    matched = episode_features.add_episode_features(
        matched, gap_s=args.gap_s, site_col=site_col)
    cands = episode_features.available_candidates(matched)
    if not cands:
        sys.exit("No candidate signal could be computed -- the table needs "
                 "start_s, recording and species columns.")

    base = {c: d for c, d in cleanup_eval.RANKING_SIGNALS.items()
            if c in matched.columns
            and pd.to_numeric(matched[c], errors="coerce").notna().any()}
    print(f"Reported signals in this table : {', '.join(base) or '(none)'}")
    print(f"Candidate temporal signals     : {', '.join(cands)}")
    print(f"Episodes at a {args.gap_s:.0f} s gap          : "
          f"{matched['episode'].nunique()} over {len(matched)} detections\n")

    out_dir = args.out or (os.path.dirname(args.matched) if args.matched
                           else args.cleanup)
    os.makedirs(out_dir, exist_ok=True)

    alone = candidates_alone(matched, cands, site_col)
    print("Each candidate on its own, both directions. Compare against the")
    print("reported signals' own average precision -- a candidate that scores")
    print("below arbitrary order at many stations is reading noise:")
    print(alone.to_string())
    alone.to_csv(os.path.join(out_dir, "rank_signals_alone.csv"))

    added = added_to_baseline(matched, cands, site_col, base)
    print("\nAdded to the four reported signals (in-sample -- read the")
    print("held-out table below before believing any positive delta):")
    print(added.to_string())
    added.to_csv(os.path.join(out_dir, "rank_signals_added.csv"))

    hs = holdout_selection(matched, cands, site_col, base,
                           max_added=args.max_added)
    if len(hs):
        print("\nForward selection, leave-one-station-out: signals chosen on the")
        print("other stations, scored on the held-out one. This is the number")
        print("that decides whether a candidate goes in the paper:")
        print(hs.to_string(index=False))
        hs.to_csv(os.path.join(out_dir, "rank_signals_holdout.csv"), index=False)

        d = pd.to_numeric(hs["delta"], errors="coerce").iloc[-1]
        print()
        if pd.isna(d):
            print("VERDICT: inconclusive -- not enough stations to hold out.")
        elif d > 0.005:
            print(f"VERDICT: the temporal signals help. Held-out mean average "
                  f"precision improves by {d:+.4f}; report it.")
        elif d > 0:
            print(f"VERDICT: a real but negligible gain ({d:+.4f} held out). "
                  f"Not worth the extra columns in the paper.")
        else:
            print(f"VERDICT: no held-out gain ({d:+.4f}). The temporal "
                  f"structure is already captured by the neighbour count; "
                  f"keep the four reported signals and say so.")

    print(f"\nWrote rank_signals_*.csv to {out_dir}/")


if __name__ == "__main__":
    main()
