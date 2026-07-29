"""
Build the manual-review queue: detections grouped into episodes and ordered
most-likely-genuine first.

This is the part of the cleanup the field evaluation found most useful, and
until now it lived only in the evaluation module, which needs manual labels to
run. That was backwards: both mechanisms are label-free, so the pipeline can
emit the queue directly and a new user gets the ordering without first having to
review anything.

What the queue does
-------------------
Two things, neither of which discards a detection or fits a parameter:

1. **Episodes.** Detections of one species close together in one recording are
   one listening decision. A reviewer who hears the sound once has heard the
   rest, so an episode of non-target noise is dismissed in a single listen
   however many detections it holds.

2. **Ordering.** Each of the signals the detector already produced is turned
   into a percentile rank *within its own station*, and the ranks are averaged.
   Ranking within the station is the point: the raw scales of these signals
   differ between sites, which is why fixed cutoffs transfer badly, but a
   percentile rank is scale-free.

Episodes are emitted most-likely-genuine first, and within an episode its
highest-ranked clip comes first and is marked ``listen_first``. That order
serves both purposes at once: genuine calls surface early, and the long runs of
non-target noise collect at the end where the reviewer meets each as a block
and can dismiss it in one go.

The queue is written in the column layout Kaleidoscope Pro expects (``INDIR``,
``IN FILE``, ``MANUAL ID``) so it can be reviewed in the same tool as
everything else, with the helper columns after them.
"""
import os

import pandas as pd

import episode_features

#: Signals to average, and whether a larger value means *more* likely genuine.
#: Kept identical to cleanup_eval.RANKING_SIGNALS -- the queue a reviewer works
#: from and the ordering the paper scores must be the same ordering.
RANKING_SIGNALS = {
    "confidence": True,
    "softmax_margin": True,
    "n_neighbours": True,
    "mahalanobis_d2": False,
}

QUEUE_FILENAME = "review_queue.csv"


def _site_col(df, site_col=None):
    if site_col:
        return site_col
    for c in ("site", "station"):
        if c in df.columns:
            return c
    return None


def _within_station_rank(values, higher_is_better, sites):
    """Percentile rank of each value inside its own station."""
    v = pd.to_numeric(values, errors="coerce")
    if not higher_is_better:
        v = -v
    return v.groupby(sites).rank(pct=True, na_option="bottom")


def rank_score(det_df, signals=None, site_col=None):
    """
    Averaged within-station percentile rank; higher means more likely genuine.

    Returns an empty Series when none of the signals is present, so a caller can
    fall back rather than emit an arbitrary order presented as a ranking.
    """
    if not len(det_df):
        return pd.Series(dtype=float)
    sc = _site_col(det_df, site_col)
    sites = det_df[sc] if sc else pd.Series("", index=det_df.index)

    signals = signals or {c: d for c, d in RANKING_SIGNALS.items()
                          if c in det_df.columns
                          and pd.to_numeric(det_df[c], errors="coerce").notna().any()}
    if not signals:
        return pd.Series(dtype=float)

    ranks = [_within_station_rank(det_df[c], higher, sites)
             for c, higher in signals.items()]
    return sum(ranks) / len(ranks)


def build(det_df, gap_s=episode_features.DEFAULT_GAP_S, signals=None,
          site_col=None):
    """
    Return the review queue as a DataFrame, in the order it should be worked.

    ``det_df`` needs ``species``, ``recording`` (or ``source_file``),
    ``start_s`` (or ``start_time``) and a station column; the ranking uses
    whichever of :data:`RANKING_SIGNALS` are present. Nothing is dropped: the
    queue holds every detection handed to it.

    Adds ``episode``, ``rank_score``, ``episode_rank``, ``listen_first`` and
    ``episode_detections``. When no ranking signal is available the episode
    grouping is still applied and the order falls back to chronological, with
    ``rank_score`` left empty -- an honest absence rather than a fake ordering.
    """
    df = _normalise_columns(det_df)
    if not len(df):
        return df.assign(episode=pd.Series(dtype="int64"),
                         rank_score=pd.Series(dtype=float),
                         listen_first=pd.Series(dtype=bool))

    df = episode_features.assign_episodes(df, gap_s=gap_s, site_col=site_col)

    score = rank_score(df, signals=signals, site_col=site_col)
    df["rank_score"] = score if len(score) else pd.NA

    # An episode is placed by its best clip, not its average: that clip is the
    # one the reviewer actually hears before deciding the episode.
    if len(score):
        df["episode_rank"] = df.groupby("episode")["rank_score"].transform("max")
        df = df.sort_values(["episode_rank", "episode", "rank_score"],
                            ascending=[False, True, False], kind="stable")
    else:
        df["episode_rank"] = pd.NA
        df = df.sort_values(["episode", "start_s"], kind="stable")

    df["episode_detections"] = df.groupby("episode")["episode"].transform("size")
    df["listen_first"] = ~df["episode"].duplicated()
    return df.reset_index(drop=True)


def _normalise_columns(det_df):
    """Accept either the detection-CSV names or the matched-table names."""
    df = det_df.copy()
    if "start_s" not in df.columns and "start_time" in df.columns:
        df["start_s"] = pd.to_numeric(df["start_time"], errors="coerce")
    if "recording" not in df.columns and "source_file" in df.columns:
        df["recording"] = (df["source_file"].astype(str)
                           .str.replace(r"\.wav$", "", regex=True))
    return df


def to_kaleidoscope(queue_df, clips_root=""):
    """
    Lay the queue out as a Kaleidoscope Pro review sheet.

    ``MANUAL ID`` is left blank for the reviewer to fill. The helper columns
    follow, so a reviewer can see which row to listen to first and how many
    detections that one listen settles, and sorting in a spreadsheet still
    works.
    """
    if not len(queue_df):
        return pd.DataFrame(columns=["INDIR", "IN FILE", "MANUAL ID"])

    sc = _site_col(queue_df) or ""
    site = queue_df[sc].astype(str) if sc else ""
    species = (queue_df["species"].astype(str) if "species" in queue_df.columns
               else "")

    if "clip_file" in queue_df.columns:
        in_file = queue_df["clip_file"].astype(str)
    else:
        in_file = _expected_clip_name(queue_df)

    out = pd.DataFrame({
        "INDIR": [os.path.join(str(clips_root), sp, st) if sc else str(clips_root)
                  for sp, st in zip(species, site)] if sc else str(clips_root),
        "IN FILE": in_file,
        "MANUAL ID": "",
    })
    for col in ("listen_first", "episode", "episode_detections", "rank_score",
                "recording", "start_s", "confidence"):
        if col in queue_df.columns:
            out[col] = queue_df[col].values
    return out


def _expected_clip_name(queue_df):
    """Reconstruct the exported clip filename written by extract_clips()."""
    rec = (queue_df["recording"].astype(str) if "recording" in queue_df.columns
           else pd.Series("", index=queue_df.index))
    start = pd.to_numeric(queue_df.get("start_s"), errors="coerce").fillna(0)
    conf = pd.to_numeric(queue_df.get("confidence"), errors="coerce").fillna(0)
    return [f"{r}__t{int(s):05d}s__conf{c:.2f}.wav"
            for r, s, c in zip(rec, start, conf)]


def save(det_df, out_dir, gap_s=episode_features.DEFAULT_GAP_S, signals=None,
         site_col=None, clips_root="", verbose=True):
    """Build the queue and write it to ``out_dir/review_queue.csv``.

    Returns the queue DataFrame. Also writes ``review_queue_episodes.csv``, one
    row per episode, which is the summary a reviewer plans from.
    """
    q = build(det_df, gap_s=gap_s, signals=signals, site_col=site_col)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, QUEUE_FILENAME)
    to_kaleidoscope(q, clips_root=clips_root).to_csv(path, index=False)

    ep = episode_summary(q)
    ep.to_csv(os.path.join(out_dir, "review_queue_episodes.csv"), index=False)

    if verbose:
        n_ep = int(q["episode"].nunique()) if len(q) else 0
        print(f'  Review queue: {len(q)} detections in {n_ep} episodes '
              f'-> {os.path.basename(path)}')
        if n_ep:
            print(f'  Listen to the {n_ep} rows marked listen_first; each '
                  f'settles its whole episode when the sound is non-target.')
    return q


def episode_summary(queue_df):
    """One row per episode, in queue order: what to listen to and what it settles."""
    if not len(queue_df):
        return pd.DataFrame()
    sc = _site_col(queue_df)
    first = queue_df[queue_df["listen_first"]] if "listen_first" in queue_df else queue_df
    cols = [c for c in (sc, "species", "recording", "episode",
                        "episode_detections", "rank_score", "start_s") if c]
    return first[[c for c in cols if c in first.columns]].reset_index(drop=True)
