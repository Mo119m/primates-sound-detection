"""
Evaluate the automatic cleanup against the manual review labels.

The manual review is performed on the *raw* detections, before the automatic
cleanup runs, so the reviewer's verdicts are ground truth for the cleanup: for
every detection we know both what a human decided (genuine call vs. noise) and
what the cleanup decided (clean vs. suspicious). Cross-tabulating the two gives
the numbers that justify the method:

  * how many confirmed false positives the cleanup removes without any
    listening (its whole purpose), and
  * how many genuine calls it costs (the price paid for that removal),
  * precision before vs. after cleanup, and
  * which of the three filters caught what.

Detections are matched on ``(species, recording, start second)``. The review
tables identify a detection through the exported clip filename::

    Cernic__20210222T053000+0100_Short-term_Makokou__01540s__conf0.980.wav
    ^species ^recording                              ^start ^confidence

and the cleanup CSVs carry ``species``, ``source_file`` and ``start_time``.
"""
import os
import re
import itertools
import pandas as pd

try:
    from . import review_import
except ImportError:  # running as a script / from a notebook
    import review_import

# recording part of the exported clip filename
_REC_RE = re.compile(r"^.+?__(?P<recording>.+)__\d+s__conf[0-9.]+\.wav$")

CLEAN = "clean"            # cleanup kept it (no filter fired)
SUSPICIOUS = "suspicious"  # cleanup flagged it (>=1 filter fired)

# Per-filter flag columns written by auto_cleanup, and their display names.
FLAG_COLUMNS = ["flag_mahal", "flag_yamnet", "flag_isolated"]
FLAG_LABELS = {"flag_mahal": "mahalanobis", "flag_yamnet": "yamnet",
               "flag_isolated": "temporal isolation"}


def _recording_of(clip_filename):
    m = _REC_RE.match(str(clip_filename))
    return m.group("recording") if m else ""


def _key(species, recording, start_s):
    return (str(species).strip(), str(recording).strip(), int(start_s))


def load_cleanup_output(cleanup_dir):
    """
    Load ``clean_detections.csv`` and ``suspicious_detections.csv`` written by
    ``auto_cleanup.run_auto_cleanup`` and return one DataFrame with a
    ``cleanup`` column ('clean' or 'suspicious').
    """
    cleanup_dir = os.path.expanduser(str(cleanup_dir))
    clean_p = os.path.join(cleanup_dir, "clean_detections.csv")
    susp_p = os.path.join(cleanup_dir, "suspicious_detections.csv")
    frames = []
    for path, verdict in ((clean_p, CLEAN), (susp_p, SUSPICIOUS)):
        if os.path.exists(path):
            df = pd.read_csv(path)
            if len(df):
                df["cleanup"] = verdict
                frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"Neither clean_detections.csv nor suspicious_detections.csv found "
            f"under {cleanup_dir!r}")
    df = pd.concat(frames, ignore_index=True)

    # 'source_file' is the recording's .wav name; strip the extension so it
    # matches the recording embedded in the review clip filename.
    if "source_file" in df.columns:
        df["recording"] = df["source_file"].astype(str).str.replace(
            r"\.wav$", "", regex=True)
    else:
        df["recording"] = ""
    return df


def match(review_df, cleanup_df, start_tolerance=1):
    """
    Join the manual verdicts with the cleanup verdicts.

    Matching is exact on ``(species, recording, start second)``; detections that
    do not match exactly are retried within +/- ``start_tolerance`` seconds to
    absorb rounding between the exported clip name and the detection table.

    Returns a DataFrame with one row per reviewed detection and the columns
    ``species, site, recording, start_s, confidence, verdict, cleanup,
    flag_reason`` where ``cleanup`` is NaN when no cleanup row matched.
    """
    rev = review_df.copy()
    rev["recording"] = rev["file"].map(_recording_of)

    index = {}
    for _, r in cleanup_df.iterrows():
        try:
            k = _key(r.get("species", ""), r.get("recording", ""),
                     round(float(r.get("start_time", -1))))
        except (TypeError, ValueError):
            continue
        index.setdefault(k, r)

    verdicts, reasons = [], []
    per_flag = {c: [] for c in FLAG_COLUMNS}
    extra = {"yamnet_top": [], "yamnet_score": [],
             "mahalanobis_d2": [], "n_neighbours": []}
    for _, r in rev.iterrows():
        hit = None
        base = _key(r["species"], r["recording"], r["start_s"])
        for delta in [0] + [d for t in range(1, int(start_tolerance) + 1)
                            for d in (-t, t)]:
            hit = index.get((base[0], base[1], base[2] + delta))
            if hit is not None:
                break
        verdicts.append(hit["cleanup"] if hit is not None else None)
        reasons.append(hit.get("flag_reason", "") if hit is not None else "")
        for col in FLAG_COLUMNS:
            per_flag[col].append(bool(hit[col]) if hit is not None
                                 and col in hit.index else None)
        # the AudioSet class YAMNet assigned, for the per-class breakdown
        for col in ("yamnet_top", "yamnet_score", "mahalanobis_d2",
                    "n_neighbours"):
            extra[col].append(hit[col] if hit is not None
                              and col in hit.index else None)

    rev["cleanup"] = verdicts
    rev["flag_reason"] = reasons
    for col in FLAG_COLUMNS:
        rev[col] = per_flag[col]
    for col, vals in extra.items():
        rev[col] = vals
    return rev


def per_filter_analysis(matched_df):
    """
    Score each filter on its own against the reviewer's labels.

    A filter is only useful if it fires on false positives more often than on
    genuine calls. ``lift`` is that ratio, P(flag | false positive) /
    P(flag | call): above 1 the filter carries signal, at or below 1 it is
    discarding real calls at least as fast as it removes noise, and using it
    lowers precision. ``precision_if_only`` is the precision that would result
    from applying this filter alone.
    """
    df = matched_df[matched_df["cleanup"].notna()]
    calls = df[df["verdict"] == "call"]
    fps = df[df["verdict"] == "false_positive"]
    n_calls, n_fps = len(calls), len(fps)

    rows = {}
    for col, label in FLAG_LABELS.items():
        if col not in df.columns or df[col].isna().all():
            continue
        c_flagged = int(calls[col].fillna(False).astype(bool).sum())
        f_flagged = int(fps[col].fillna(False).astype(bool).sum())
        p_call = c_flagged / n_calls if n_calls else 0.0
        p_fp = f_flagged / n_fps if n_fps else 0.0
        kept_calls = n_calls - c_flagged
        kept_total = kept_calls + (n_fps - f_flagged)
        rows[label] = {
            "flagged_total": c_flagged + f_flagged,
            "calls_flagged": c_flagged,
            "fps_flagged": f_flagged,
            "pct_of_calls": round(100 * p_call, 1),
            "pct_of_fps": round(100 * p_fp, 1),
            "lift": round(p_fp / p_call, 2) if p_call else None,
            "precision_if_only": (round(kept_calls / kept_total, 4)
                                  if kept_total else None),
        }
    return pd.DataFrame(rows).T


def filter_combination_analysis(matched_df):
    """
    Score every subset of the filters, from the per-detection flags.

    A detection is discarded when any filter in the subset fires, so each
    subset's outcome is already determined by the flags recorded in a single
    run -- the ablation needs no re-run of the model. Sorted by the resulting
    precision, best first.
    """
    df = matched_df[matched_df["cleanup"].notna()]
    present = [c for c in FLAG_COLUMNS
               if c in df.columns and not df[c].isna().all()]
    if not present:
        return pd.DataFrame()

    calls = df["verdict"] == "call"
    n_calls = int(calls.sum())
    n_fps = int((df["verdict"] == "false_positive").sum())
    total = len(df)

    rows = {}
    for size in range(len(present) + 1):
        for combo in itertools.combinations(present, size):
            if combo:
                flagged = df[list(combo)].fillna(False).astype(bool).any(axis=1)
            else:
                flagged = pd.Series(False, index=df.index)
            kept = ~flagged
            kept_calls = int((kept & calls).sum())
            kept_total = int(kept.sum())
            name = " + ".join(FLAG_LABELS[c] for c in combo) or "(no filter)"
            rows[name] = {
                "calls_kept": kept_calls,
                "calls_lost": n_calls - kept_calls,
                "fps_removed": n_fps - (kept_total - kept_calls),
                "reviewed_after": kept_total,
                "pct_removed": round(100 * (total - kept_total) / total, 1),
                "precision": (round(kept_calls / kept_total, 4)
                              if kept_total else None),
            }
    out = pd.DataFrame(rows).T
    return out.sort_values("precision", ascending=False)


def _outcome(df, flagged, n_calls, n_fps, total):
    """Shared tally for a boolean discard mask."""
    kept = ~flagged
    calls = df["verdict"] == "call"
    kept_calls = int((kept & calls).sum())
    kept_total = int(kept.sum())
    return {
        "calls_kept": kept_calls,
        "calls_lost": n_calls - kept_calls,
        "fps_removed": n_fps - (kept_total - kept_calls),
        "reviewed_after": kept_total,
        "pct_removed": round(100 * (total - kept_total) / total, 1),
        "precision": round(kept_calls / kept_total, 4) if kept_total else None,
    }


def vote_analysis(matched_df):
    """
    Require agreement between filters before discarding a detection.

    The pipeline currently discards on a single flag. Demanding two or three
    filters agree trades away removals for safety: fewer false positives go, but
    fewer genuine calls are lost with them. This reports each voting threshold
    so the trade can be read off directly.
    """
    df = matched_df[matched_df["cleanup"].notna()]
    present = [c for c in FLAG_COLUMNS
               if c in df.columns and not df[c].isna().all()]
    if not present:
        return pd.DataFrame()

    n_flags = df[present].fillna(False).astype(bool).sum(axis=1)
    n_calls = int((df["verdict"] == "call").sum())
    n_fps = int((df["verdict"] == "false_positive").sum())
    total = len(df)

    rows = {}
    for k in range(1, len(present) + 1):
        rows[f">= {k} filter(s) agree"] = _outcome(df, n_flags >= k, n_calls,
                                                   n_fps, total)
    rows["no cleanup"] = _outcome(df, pd.Series(False, index=df.index),
                                  n_calls, n_fps, total)
    return pd.DataFrame(rows).T


def yamnet_score_sweep(matched_df, thresholds=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                                               0.6, 0.7, 0.8)):
    """
    Require a minimum YAMNet confidence before its verdict is trusted.

    YAMNet always returns a top class, so on unfamiliar forest audio it labels
    every clip whether or not it recognises the sound; the pipeline acts on that
    label regardless of the score behind it. Restricting the filter to
    confident predictions is the natural way to give it an "unsure, do not
    flag" option. Each row combines the other filters with YAMNet restricted to
    predictions scoring at least the threshold.
    """
    df = matched_df[matched_df["cleanup"].notna()]
    if "yamnet_score" not in df.columns or df["yamnet_score"].isna().all():
        return pd.DataFrame()

    others = [c for c in ("flag_mahal", "flag_isolated")
              if c in df.columns and not df[c].isna().all()]
    base = (df[others].fillna(False).astype(bool).any(axis=1) if others
            else pd.Series(False, index=df.index))
    yam = df["flag_yamnet"].fillna(False).astype(bool)
    score = pd.to_numeric(df["yamnet_score"], errors="coerce").fillna(0.0)

    n_calls = int((df["verdict"] == "call").sum())
    n_fps = int((df["verdict"] == "false_positive").sum())
    total = len(df)

    rows = {"other filters only": _outcome(df, base, n_calls, n_fps, total)}
    for t in thresholds:
        flagged = base | (yam & (score >= t))
        rows[f"+ yamnet (score >= {t:.1f})"] = _outcome(df, flagged, n_calls,
                                                        n_fps, total)
    return pd.DataFrame(rows).T


def signal_sweep(matched_df, column, flag_when="high", n_steps=15):
    """
    Sweep one continuous signal's cutoff and report the outcome at each.

    The filters threshold continuous quantities that the run already records --
    Mahalanobis distance, neighbour count, detector confidence -- at values
    chosen a priori. Sweeping the cutoff against the reviewed labels shows what
    those choices cost and where the useful operating points are.

    ``flag_when`` is "high" when large values are suspicious (Mahalanobis
    distance) and "low" when small ones are (neighbour count, confidence).
    """
    df = matched_df[matched_df["cleanup"].notna()].copy()
    if column not in df.columns:
        return pd.DataFrame()
    vals = pd.to_numeric(df[column], errors="coerce")
    if vals.notna().sum() == 0:
        return pd.DataFrame()

    n_calls = int((df["verdict"] == "call").sum())
    n_fps = int((df["verdict"] == "false_positive").sum())
    total = len(df)

    qs = [i / (n_steps + 1) for i in range(1, n_steps + 1)]
    cuts = sorted({round(float(vals.quantile(q)), 4) for q in qs})

    rows = {}
    for c in cuts:
        flagged = (vals >= c) if flag_when == "high" else (vals <= c)
        flagged = flagged.fillna(False)
        rows[f"{column} {'>=' if flag_when == 'high' else '<='} {c:g}"] = (
            _outcome(df, flagged, n_calls, n_fps, total))
    out = pd.DataFrame(rows).T
    return out.sort_values(["precision", "calls_kept"], ascending=False)


def optimize_thresholds(matched_df, min_call_retention=0.95, n_steps=12):
    """
    Search Mahalanobis and neighbour cutoffs jointly for the best precision
    that still keeps at least ``min_call_retention`` of the genuine calls.

    The two filters are applied together (a detection is discarded if either
    fires), so their cutoffs interact and are searched as a grid rather than
    tuned one at a time. The retention floor is what keeps the search from
    "improving" precision by discarding most of the data.
    """
    df = matched_df[matched_df["cleanup"].notna()].copy()
    d2 = pd.to_numeric(df.get("mahalanobis_d2"), errors="coerce")
    nb = pd.to_numeric(df.get("n_neighbours"), errors="coerce")
    if d2 is None or nb is None or d2.notna().sum() == 0 or nb.notna().sum() == 0:
        return pd.DataFrame()

    n_calls = int((df["verdict"] == "call").sum())
    n_fps = int((df["verdict"] == "false_positive").sum())
    total = len(df)
    floor = min_call_retention * n_calls

    qs = [i / (n_steps + 1) for i in range(1, n_steps + 1)]
    d2_cuts = sorted({float(d2.quantile(q)) for q in qs}) + [float("inf")]
    nb_cuts = sorted({int(nb.quantile(q)) for q in qs}) + [-1]

    rows = {}
    for dc in d2_cuts:
        for nc in nb_cuts:
            flagged = ((d2 >= dc).fillna(False)) | ((nb <= nc).fillna(False))
            res = _outcome(df, flagged, n_calls, n_fps, total)
            if res["calls_kept"] < floor:
                continue
            label = (f"mahal >= {dc:.0f}" if dc != float("inf") else "mahal off")
            label += f", neighbours <= {nc}" if nc >= 0 else ", isolation off"
            rows[label] = res
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).T
    return out.sort_values(["precision", "calls_kept"], ascending=False).head(12)


def station_holdout_sweep(matched_df, column="confidence", flag_when="low",
                          min_call_retention=0.60, n_steps=25):
    """
    Choose a cutoff on one set of stations and report it on the others.

    A cutoff picked on the same detections it is then scored on is fitted to
    them, and the resulting precision is not an estimate of anything. Stations
    are the natural split here because each one has its own soundscape: sorting
    them by name and alternating assigns roughly half the data to tuning and
    half to evaluation while keeping whole stations intact.

    The cutoff maximises precision on the tuning half subject to keeping
    ``min_call_retention`` of its genuine calls; the row reported for the
    evaluation half is what that same cutoff achieves on stations that had no
    part in choosing it.
    """
    df = matched_df[matched_df["cleanup"].notna()].copy()
    if column not in df.columns or "site" not in df.columns:
        return pd.DataFrame()
    vals = pd.to_numeric(df[column], errors="coerce")
    if vals.notna().sum() == 0:
        return pd.DataFrame()

    sites = sorted(df["site"].dropna().unique())
    if len(sites) < 4:
        return pd.DataFrame()
    tune_sites = set(sites[0::2])
    tune = df[df["site"].isin(tune_sites)]
    test = df[~df["site"].isin(tune_sites)]
    if not len(tune) or not len(test):
        return pd.DataFrame()

    def score(sub, cutoff):
        v = pd.to_numeric(sub[column], errors="coerce")
        flagged = (v <= cutoff) if flag_when == "low" else (v >= cutoff)
        flagged = flagged.fillna(False)
        n_calls = int((sub["verdict"] == "call").sum())
        n_fps = int((sub["verdict"] == "false_positive").sum())
        return _outcome(sub, flagged, n_calls, n_fps, len(sub))

    qs = [i / (n_steps + 1) for i in range(1, n_steps + 1)]
    cuts = sorted({float(vals.quantile(q)) for q in qs})
    tune_calls = int((tune["verdict"] == "call").sum())

    best, best_cut = None, None
    for c in cuts:
        r = score(tune, c)
        if r["calls_kept"] < min_call_retention * tune_calls:
            continue
        if best is None or (r["precision"] or 0) > (best["precision"] or 0):
            best, best_cut = r, c
    if best is None:
        return pd.DataFrame()

    rows = {
        f"tuning stations ({len(tune_sites)}), cutoff {best_cut:.4g}": best,
        f"held-out stations ({len(sites) - len(tune_sites)}), same cutoff":
            score(test, best_cut),
        "held-out stations, no cleanup":
            score(test, -1e18 if flag_when == "low" else 1e18),
    }
    return pd.DataFrame(rows).T


def confidence_baseline(matched_df, filter_cols=("flag_mahal", "flag_isolated")):
    """
    Compare the cleanup against simply discarding the least confident detections.

    The filters are only worth their complexity if they beat the trivial
    alternative already available from the detector: drop the lowest-confidence
    detections until as many are gone as the cleanup would remove. Both rows
    discard the same number of detections, so their precision is directly
    comparable, and ``advantage`` is the cleanup's precision minus the
    baseline's -- positive means the filters find false positives that model
    confidence alone does not.
    """
    df = matched_df[matched_df["cleanup"].notna()].copy()
    cols = [c for c in filter_cols if c in df.columns and not df[c].isna().all()]
    if not cols or "confidence" not in df.columns:
        return pd.DataFrame()

    calls = df["verdict"] == "call"
    n_calls = int(calls.sum())
    n_fps = int((df["verdict"] == "false_positive").sum())
    total = len(df)

    flagged = df[cols].fillna(False).astype(bool).any(axis=1)
    n_removed = int(flagged.sum())

    # Same budget, spent on the least confident detections instead.
    conf = pd.to_numeric(df["confidence"], errors="coerce")
    order = conf.sort_values(kind="stable").index[:n_removed]
    by_conf = pd.Series(False, index=df.index)
    by_conf.loc[order] = True

    name = " + ".join(FLAG_LABELS[c] for c in cols)
    rows = {
        f"cleanup ({name})": _outcome(df, flagged, n_calls, n_fps, total),
        f"lowest confidence ({n_removed} clips)":
            _outcome(df, by_conf, n_calls, n_fps, total),
    }
    out = pd.DataFrame(rows).T
    out["advantage"] = [round(out.iloc[0]["precision"] - out.iloc[1]["precision"], 4),
                        None]
    return out


def yamnet_class_analysis(matched_df, min_count=10):
    """
    Break the YAMNet filter down by the AudioSet class it assigned.

    The filter treats a fixed set of classes as non-primate. Whether that is
    right is an empirical question per class: a class that lands mostly on
    false positives is worth flagging, while one that lands mostly on genuine
    calls is mislabelling the target species and should be removed from the
    suspicious set. Grouping the reviewed detections by ``yamnet_top`` shows
    which is which.

    Classes seen fewer than ``min_count`` times are dropped as too rare to act
    on. Sorted by the share of genuine calls, so the classes doing the damage
    come first.
    """
    df = matched_df[matched_df["cleanup"].notna()]
    if "yamnet_top" not in df.columns or df["yamnet_top"].isna().all():
        return pd.DataFrame()

    rows = {}
    for cls, sub in df.groupby("yamnet_top"):
        n = len(sub)
        if n < min_count:
            continue
        n_calls = int((sub["verdict"] == "call").sum())
        n_fps = int((sub["verdict"] == "false_positive").sum())
        flagged = bool(sub["flag_yamnet"].fillna(False).astype(bool).any())
        rows[cls] = {
            "n": n,
            "calls": n_calls,
            "false_positives": n_fps,
            "pct_calls": round(100 * n_calls / n, 1) if n else None,
            "in_suspicious_set": flagged,
        }
    out = pd.DataFrame(rows).T
    if len(out):
        out = out.sort_values("pct_calls", ascending=False)
    return out


def evaluate(matched_df):
    """
    Cross-tabulate manual verdict against cleanup verdict.

    Returns a dict with:
      * ``confusion``  -- DataFrame: rows = manual verdict, cols = cleanup verdict
      * ``overall``    -- dict of headline numbers (see keys below)
      * ``per_species``-- DataFrame of the same numbers per species
      * ``unmatched``  -- number of reviewed detections with no cleanup row

    Headline numbers, all computed on detections that matched:
      detections, calls, false_positives,
      fp_removed, fp_removed_pct   -- false positives the cleanup flagged
      calls_kept, calls_kept_pct   -- genuine calls the cleanup left clean
      calls_lost                   -- genuine calls the cleanup flagged
      precision_before             -- calls / (calls + false positives)
      precision_after              -- calls kept / all detections left clean
      listening_reduction_pct      -- share of detections removed from review
    """
    df = matched_df
    matched = df[df["cleanup"].notna()]
    unmatched = int(df["cleanup"].isna().sum())

    confusion = (pd.crosstab(matched["verdict"], matched["cleanup"])
                 if len(matched) else pd.DataFrame())

    def block(sub):
        calls = sub[sub["verdict"] == "call"]
        fps = sub[sub["verdict"] == "false_positive"]
        n_calls, n_fp = len(calls), len(fps)
        fp_removed = int((fps["cleanup"] == SUSPICIOUS).sum())
        calls_kept = int((calls["cleanup"] == CLEAN).sum())
        calls_lost = n_calls - calls_kept
        left_clean = int((sub["cleanup"] == CLEAN).sum())
        total = len(sub)
        return {
            "detections": total,
            "calls": n_calls,
            "false_positives": n_fp,
            "fp_removed": fp_removed,
            "fp_removed_pct": round(100 * fp_removed / n_fp, 1) if n_fp else None,
            "calls_kept": calls_kept,
            "calls_kept_pct": round(100 * calls_kept / n_calls, 1) if n_calls else None,
            "calls_lost": calls_lost,
            "precision_before": round(n_calls / total, 4) if total else None,
            "precision_after": round(calls_kept / left_clean, 4) if left_clean else None,
            "listening_reduction_pct": (round(100 * (total - left_clean) / total, 1)
                                        if total else None),
        }

    overall = block(matched)
    per_species = pd.DataFrame(
        {sp: block(sub) for sp, sub in matched.groupby("species")}
    ).T if len(matched) else pd.DataFrame()

    # Which filter caught each removed false positive.
    removed = matched[(matched["verdict"] == "false_positive") &
                      (matched["cleanup"] == SUSPICIOUS)]
    by_filter = {}
    for name, needle in (("mahalanobis", "mahal"), ("yamnet", "yamnet"),
                         ("temporal isolation", "isolat")):
        by_filter[name] = int(removed["flag_reason"].astype(str)
                              .str.contains(needle, case=False, na=False).sum())

    return {"confusion": confusion, "overall": overall,
            "per_species": per_species, "unmatched": unmatched,
            "fp_removed_by_filter": by_filter}


# The two ways the cleanup and the reviewer can disagree.
WRONGLY_FLAGGED = "wrongly_flagged"   # reviewer: call,  cleanup: suspicious
MISSED = "missed"                     # reviewer: noise, cleanup: clean


def disagreements(matched_df):
    """
    Rows where the cleanup and the reviewer disagree, tagged by which way.

    ``wrongly_flagged`` is the expensive error: a genuine call the cleanup would
    have discarded. ``missed`` is a false positive the cleanup let through.

    Listening to these explains *why* the cleanup fails and what to mine as hard
    negatives. Do not use it to revise the labels only where the cleanup
    disagreed -- correcting errors in one direction only biases the ground truth
    in the cleanup's favour and inflates the reported precision.
    """
    df = matched_df[matched_df["cleanup"].notna()].copy()
    kind = []
    for _, r in df.iterrows():
        if r["verdict"] == "call" and r["cleanup"] == SUSPICIOUS:
            kind.append(WRONGLY_FLAGGED)
        elif r["verdict"] == "false_positive" and r["cleanup"] == CLEAN:
            kind.append(MISSED)
        else:
            kind.append("")
    df["disagreement"] = kind
    out = df[df["disagreement"] != ""].copy()
    cols = [c for c in ["disagreement", "species", "site", "recording", "start_s",
                        "confidence", "verdict", "cleanup", "flag_reason", "file"]
            if c in out.columns]
    return out[cols].sort_values(["disagreement", "species", "site", "start_s"],
                                 kind="stable").reset_index(drop=True)


def report_text(matched_df):
    """Plain-text report of the cleanup's effect, ready for the manuscript."""
    e = evaluate(matched_df)
    o = e["overall"]
    L = ["AUTOMATIC CLEANUP vs MANUAL REVIEW (ground truth)",
         "=" * 52,
         f"Reviewed detections matched to cleanup : {o['detections']}"]
    if e["unmatched"]:
        L.append(f"  (!) {e['unmatched']} reviewed detections had no cleanup row "
                 f"and are excluded")
    L += ["",
          f"Confirmed calls    : {o['calls']}",
          f"False positives    : {o['false_positives']}",
          "",
          f"False positives removed by cleanup : {o['fp_removed']}"
          f" ({o['fp_removed_pct']}%)",
          f"Genuine calls retained             : {o['calls_kept']}"
          f" ({o['calls_kept_pct']}%)",
          f"Genuine calls lost to cleanup      : {o['calls_lost']}",
          "",
          f"Precision before cleanup : "
          f"{o['precision_before']*100:.1f}%" if o['precision_before'] is not None else "",
          f"Precision after cleanup  : "
          f"{o['precision_after']*100:.1f}%" if o['precision_after'] is not None else "",
          f"Detections removed from manual review : {o['listening_reduction_pct']}%",
          "",
          "False positives removed, by filter (a clip may trip several):"]
    for k, v in e["fp_removed_by_filter"].items():
        L.append(f"  {k:20s}: {v}")
    return "\n".join([x for x in L if x != ""])


def run(review_source, cleanup_dir, blank_is_confirmed=True, start_tolerance=1):
    """Convenience: load review CSVs + cleanup output, match, and return both
    the matched DataFrame and the evaluation dict."""
    rev = review_import.load_review_dir(review_source,
                                        blank_is_confirmed=blank_is_confirmed)
    cln = load_cleanup_output(cleanup_dir)
    matched = match(rev, cln, start_tolerance=start_tolerance)
    return matched, evaluate(matched)
