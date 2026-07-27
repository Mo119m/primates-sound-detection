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
import pandas as pd

try:
    from . import review_import
except ImportError:  # running as a script / from a notebook
    import review_import

# recording part of the exported clip filename
_REC_RE = re.compile(r"^.+?__(?P<recording>.+)__\d+s__conf[0-9.]+\.wav$")

CLEAN = "clean"            # cleanup kept it (no filter fired)
SUSPICIOUS = "suspicious"  # cleanup flagged it (>=1 filter fired)


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

    rev["cleanup"] = verdicts
    rev["flag_reason"] = reasons
    return rev


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
