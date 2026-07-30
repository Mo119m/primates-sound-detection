"""
Decide whether a retrained model is worth a full detection run.

Re-running detection over every continuous recording is the expensive step, and
it is wasted if the new model is no better. It does not have to be run to find
out. The manual review left behind something more useful than its own result:
6 189 exported clips whose true labels are known. Any candidate model can be
run over those clips in minutes, and that answers the question the full run
would answer about *known* errors --- does the new model stop firing on the
false positives, and does it still fire on the genuine calls.

What this measures, and what it does not
----------------------------------------
The reviewed clips are the windows the **old** model reported. Scoring a new
model on them therefore measures:

* how many known false positives it no longer emits (precision gain), and
* how many known genuine calls it still emits (the cost of that gain).

It cannot measure false positives the new model raises on audio the old model
was silent on, because no clip exists there, nor calls that both models miss.
Those need a pilot detection run on a sample of continuous audio -- a much
smaller job than the full run, and the reason :func:`gate` reports a verdict of
"pilot" rather than "ship".

The transfer check is the one that matters
-------------------------------------------
Every improvement attempted on this dataset that was fitted to particular
stations failed to carry to the others. Hard-negative mining has exactly that
shape: negatives mined at one station can teach the model that station's noise
and nothing more. :func:`gate` therefore requires the gain to hold at stations
whose clips were *not* mined, and reports the two numbers separately. A
candidate that improves only where its training data came from has not been
shown to work.
"""
import numpy as np
import pandas as pd

#: A candidate must keep at least this share of the genuine calls the old model
#: found. Set high deliberately: precision is cheap to buy by discarding calls,
#: and a detector that loses calls fails at the thing it exists for.
DEFAULT_MIN_CALL_RETENTION = 0.95

#: ... and must improve precision by at least this much at unmined stations to
#: be worth the cost of a full run.
DEFAULT_MIN_PRECISION_GAIN = 0.05


def _site_col(df, site_col=None):
    if site_col:
        return site_col
    return "site" if "site" in df.columns else "station"


def apply_candidate(reviewed, predictions, species=None, threshold=None,
                    key_cols=("recording", "start_s")):
    """
    Join a candidate model's per-clip output onto the reviewed detections.

    ``predictions`` needs the same keys as ``reviewed`` plus ``pred_species``
    and ``pred_confidence``. A row is marked ``kept`` when the candidate would
    still emit it: it predicts the target species with at least ``threshold``
    confidence. Clips with no prediction are dropped, with the count returned,
    rather than silently assumed either way.
    """
    rev = reviewed.copy()
    pred = predictions.copy()
    for c in ("start_s", "pred_confidence"):
        if c in pred.columns:
            pred[c] = pd.to_numeric(pred[c], errors="coerce")
    if "start_s" in rev.columns:
        rev["start_s"] = pd.to_numeric(rev["start_s"], errors="coerce")

    keys = [c for c in key_cols if c in rev.columns and c in pred.columns]
    if not keys:
        raise ValueError("reviewed and predictions share no join key "
                         f"(looked for {list(key_cols)})")
    # Round the time key so a sub-second difference in how the two tables were
    # written does not silently drop every row.
    for f in (rev, pred):
        if "start_s" in f.columns:
            f["_t"] = f["start_s"].round(0)
    join_on = [k if k != "start_s" else "_t" for k in keys]

    cols = join_on + [c for c in ("pred_species", "pred_confidence")
                      if c in pred.columns]
    merged = rev.merge(pred[cols].drop_duplicates(join_on), on=join_on,
                       how="left")
    n_missing = int(merged["pred_species"].isna().sum())
    merged = merged[merged["pred_species"].notna()].copy()

    target = species or (rev["species"].mode().iloc[0]
                         if "species" in rev.columns and len(rev) else None)
    is_target = merged["pred_species"].astype(str) == str(target)
    if threshold is not None:
        is_target &= merged["pred_confidence"].fillna(0.0) >= float(threshold)
    merged["kept"] = is_target
    return merged, n_missing


def score(scored_df, site_col=None):
    """
    Precision before and after, and what it cost, over the reviewed clips.

    ``scored_df`` is the output of :func:`apply_candidate`: it carries the
    reviewer's ``verdict`` and the candidate's ``kept``.
    """
    df = scored_df
    if not len(df):
        return {}
    calls = df["verdict"] == "call"
    fps = df["verdict"] == "false_positive"
    kept = df["kept"].astype(bool)

    n_calls, n_fps = int(calls.sum()), int(fps.sum())
    kept_calls = int((calls & kept).sum())
    kept_fps = int((fps & kept).sum())
    n_kept = kept_calls + kept_fps

    return {
        "clips": len(df),
        "calls": n_calls,
        "false_positives": n_fps,
        "calls_kept": kept_calls,
        "call_retention": round(kept_calls / n_calls, 4) if n_calls else None,
        "fps_removed": n_fps - kept_fps,
        "fp_removal": round((n_fps - kept_fps) / n_fps, 4) if n_fps else None,
        "precision_before": round(n_calls / len(df), 4),
        "precision_after": round(kept_calls / n_kept, 4) if n_kept else None,
        "precision_gain": (round(kept_calls / n_kept - n_calls / len(df), 4)
                           if n_kept else None),
        "detections_after": n_kept,
    }


def per_station(scored_df, site_col=None):
    """The same numbers per station -- where a mined-station-only gain shows up."""
    sc = _site_col(scored_df, site_col)
    if sc not in scored_df.columns:
        return pd.DataFrame()
    rows = []
    for st, sub in scored_df.groupby(sc, sort=True):
        s = score(sub)
        if s:
            rows.append({sc: str(st), **s})
    return pd.DataFrame(rows)


def gate(scored_df, mined_from=(), site_col=None,
         min_call_retention=DEFAULT_MIN_CALL_RETENTION,
         min_precision_gain=DEFAULT_MIN_PRECISION_GAIN):
    """
    Should the full detection run be spent on this candidate?

    Returns a dict with the overall score, the score restricted to stations
    whose clips were not mined, and a ``verdict`` of:

    * ``"reject"``  -- it loses too many genuine calls, or gains nothing where it
      was not trained;
    * ``"pilot"``   -- it passes on the reviewed clips, which is as far as those
      clips can take it; run detection on a sample of continuous audio to check
      for new false positives before committing to everything;
    * ``"mined-only"`` -- it improves, but only at the stations it was mined
      from, which is the failure this whole evaluation exists to catch.

    There is deliberately no ``"ship"``. The reviewed clips cannot rule out new
    false positives on audio the old model passed over, so nothing measured here
    justifies skipping the pilot.
    """
    sc = _site_col(scored_df, site_col)
    overall = score(scored_df)
    if not overall:
        return {"verdict": "reject", "reason": "nothing to score"}

    mined = {str(s) for s in mined_from}
    unmined = scored_df
    if mined and sc in scored_df.columns:
        unmined = scored_df[~scored_df[sc].astype(str).isin(mined)]
    held = score(unmined) if len(unmined) else {}
    mined_score = (score(scored_df[scored_df[sc].astype(str).isin(mined)])
                   if mined and sc in scored_df.columns else {})

    retention = held.get("call_retention", overall.get("call_retention"))
    gain = held.get("precision_gain", overall.get("precision_gain"))

    if retention is not None and retention < min_call_retention:
        verdict = "reject"
        reason = (f"keeps only {retention:.1%} of genuine calls at unmined "
                  f"stations, below the {min_call_retention:.0%} floor. "
                  f"Precision bought by losing calls is not an improvement.")
    elif gain is None:
        verdict = "reject"
        reason = "no precision could be computed"
    elif gain < min_precision_gain:
        if mined_score.get("precision_gain", 0) >= min_precision_gain:
            verdict = "mined-only"
            reason = (f"precision improves by "
                      f"{mined_score['precision_gain']:+.1%} at the mined "
                      f"stations but only {gain:+.1%} elsewhere. The model has "
                      f"learned those stations' noise, not the problem. Mine "
                      f"from more stations, or accept it as a local fix.")
        else:
            verdict = "reject"
            reason = (f"precision gain of {gain:+.1%} at unmined stations is "
                      f"below the {min_precision_gain:.0%} bar; a full run "
                      f"would not pay for itself.")
    else:
        verdict = "pilot"
        reason = (f"gains {gain:+.1%} precision at unmined stations while "
                  f"keeping {retention:.1%} of genuine calls. Run detection on "
                  f"a sample of continuous audio next: these clips cannot show "
                  f"false positives the old model never reported.")

    return {"verdict": verdict, "reason": reason, "overall": overall,
            "unmined": held, "mined": mined_score,
            "mined_from": sorted(mined)}


def compare(scored_a, scored_b, labels=("current", "candidate")):
    """Two candidates side by side on the same reviewed clips."""
    rows = {}
    for label, df in zip(labels, (scored_a, scored_b)):
        s = score(df)
        if s:
            rows[label] = s
    return pd.DataFrame(rows).T


def lost_calls(scored_df, site_col=None):
    """
    The genuine calls the candidate would no longer emit.

    Worth listening to before accepting a candidate: if they are all faint or
    truncated, the trade is one kind of call for a lot of noise, which may be
    acceptable; if they are ordinary clear calls, it is not.
    """
    df = scored_df[(scored_df["verdict"] == "call") & (~scored_df["kept"])]
    sc = _site_col(scored_df, site_col)
    cols = [c for c in (sc, "species", "recording", "start_s", "confidence",
                        "pred_species", "pred_confidence") if c in df.columns]
    return df[cols].sort_values("pred_confidence", ascending=False) \
        if "pred_confidence" in df.columns else df[cols]


def summarise_text(g):
    """A short report of a gate result, for printing."""
    if not g or "overall" not in g:
        return f"VERDICT: {g.get('verdict', 'reject')} -- {g.get('reason', '')}"
    o, h = g["overall"], g.get("unmined") or {}
    lines = [
        f"Reviewed clips scored : {o['clips']} "
        f"({o['calls']} calls, {o['false_positives']} false positives)",
        f"Precision             : {o['precision_before']:.1%} -> "
        f"{o['precision_after']:.1%}  ({o['precision_gain']:+.1%})",
        f"Genuine calls kept    : {o['calls_kept']}/{o['calls']} "
        f"({o['call_retention']:.1%})",
        f"False positives gone  : {o['fps_removed']}/{o['false_positives']} "
        f"({o['fp_removal']:.1%})",
    ]
    if g.get("mined_from"):
        lines.append("")
        lines.append(f"Mined from            : {', '.join(g['mined_from'])}")
        if g.get("mined"):
            m = g["mined"]
            lines.append(f"  at those stations   : {m['precision_before']:.1%} -> "
                         f"{m['precision_after']:.1%} ({m['precision_gain']:+.1%})")
        if h:
            lines.append(f"  ELSEWHERE           : {h['precision_before']:.1%} -> "
                         f"{h['precision_after']:.1%} ({h['precision_gain']:+.1%}), "
                         f"keeping {h['call_retention']:.1%} of calls")
            lines.append("  (the second line is the one that decides it)")
    lines += ["", f"VERDICT: {g['verdict'].upper()} -- {g['reason']}"]
    return "\n".join(lines)
