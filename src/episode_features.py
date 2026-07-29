"""
Temporal features derived from a detection table alone -- no labels, no model.

The four signals the cleanup ranks on are all *per-clip*: they describe one
2 s window in isolation. The field review showed that the case they all fail on
is a station overrun by an untrained species, and the thing that distinguishes
that case is not any single clip but the shape of the run the clip sits in: the
intruder calls in long, dense bouts. That structure is visible in the detection
timestamps and nowhere in the per-clip signals, which is the argument for
deriving it here.

Everything in this module is computed from ``species``, ``recording``,
``start_s`` and a station column. That matters twice over: it needs no manual
review, so the pipeline can emit it; and it needs no audio and no model, so it
can be recomputed from a detection CSV that already exists.

Whether any of these features actually *helps* is a separate question, answered
by scripts/rank_signals_experiment.py rather than assumed here. In particular
the useful direction of ``episode_size`` is genuinely unclear in advance:
bout-calling primates cluster too, so a large episode is not obviously more
suspicious than a small one. The experiment reports both directions.
"""
import numpy as np
import pandas as pd

# The gap that separates two listening episodes, matching cleanup_eval.episodes.
DEFAULT_GAP_S = 300.0

#: Derived columns and the direction that *should* mean "more likely genuine",
#: as a hypothesis to be tested rather than a setting to be trusted. See the
#: module docstring: these are candidates, not part of the reported ranking.
CANDIDATE_SIGNALS = {
    # A dense machine-gun run is more like a running intruder than a bout.
    "episode_rate": False,
    # ... and the same thing read as spacing rather than density.
    "episode_mean_gap_s": True,
    # A genuinely isolated detection contradicts bout-calling.
    "gap_to_nearest_s": False,
    # Direction unclear in advance -- primates cluster as well.
    "episode_size": False,
    "episode_span_s": False,
    # Where in its own run the detection sits; the middle of a long run is the
    # least informative place for a reviewer to start.
    "episode_position": False,
}


def _site_col(df, site_col=None):
    if site_col:
        return site_col
    return "site" if "site" in df.columns else "station"


def assign_episodes(det_df, gap_s=DEFAULT_GAP_S, site_col=None):
    """
    Label each detection with the listening episode it belongs to.

    Detections of one species in one recording separated by less than ``gap_s``
    are one episode: a reviewer who listens to one of them has heard the sound
    the others are. This is the same grouping cleanup_eval.episodes uses, split
    out here so it can run before any review exists.

    Returns a copy of ``det_df`` with an added integer ``episode`` column, in
    the input's original row order.
    """
    df = det_df.copy()
    if not len(df) or "start_s" not in df.columns:
        df["episode"] = pd.Series(dtype="int64")
        return df

    sc = _site_col(df, site_col)
    keys = [c for c in (sc, "recording", "species") if c in df.columns]
    order = df.sort_values(keys + ["start_s"], kind="stable").index

    start = pd.to_numeric(df.loc[order, "start_s"], errors="coerce")
    if keys:
        block = df.loc[order, keys]
        new_group = block.ne(block.shift()).any(axis=1)
    else:
        new_group = pd.Series(False, index=order)
        if len(order):
            new_group.iloc[0] = True
    episode = (new_group | (start.diff() > gap_s)).cumsum()

    df["episode"] = episode.reindex(df.index).astype("int64")
    return df


def add_episode_features(det_df, gap_s=DEFAULT_GAP_S, site_col=None):
    """
    Add the derived temporal features to a detection table.

    Adds ``episode`` plus every key of :data:`CANDIDATE_SIGNALS`. Columns that
    cannot be computed (a table with no ``start_s``) are left absent rather than
    filled with a guess, so a caller can test for them.

    Singleton episodes have no internal spacing to measure. Rather than invent
    one, ``episode_mean_gap_s`` is set to ``gap_s`` for them -- the largest
    spacing the grouping would still have called one episode -- which keeps them
    at the sparse end of the scale where a lone detection belongs, without
    creating an infinity that would distort a percentile rank.
    """
    df = assign_episodes(det_df, gap_s=gap_s, site_col=site_col)
    if "episode" not in df.columns or not len(df) or "start_s" not in df.columns:
        return df

    start = pd.to_numeric(df["start_s"], errors="coerce")
    grp = start.groupby(df["episode"])

    size = grp.transform("size").astype(float)
    span = (grp.transform("max") - grp.transform("min")).astype(float)

    df["episode_size"] = size
    df["episode_span_s"] = span
    # Detections per second of the run. +1 keeps a zero-span episode finite.
    df["episode_rate"] = size / (span + 1.0)
    # Mean spacing between consecutive detections inside the run.
    df["episode_mean_gap_s"] = np.where(size > 1, span / (size - 1.0), float(gap_s))
    # 0 at the start of the run, 1 at its end; 0 for a singleton.
    df["episode_position"] = np.where(span > 0,
                                      (start - grp.transform("min")) / span, 0.0)

    df["gap_to_nearest_s"] = _gap_to_nearest(df, site_col=site_col)
    return df


def _gap_to_nearest(det_df, site_col=None):
    """
    Seconds to the nearest other detection of the same species in the same
    recording -- the continuous form of the temporal-isolation filter.

    The filter asks a yes/no question ("any neighbour within 30 s?"), which
    throws away how isolated an isolated detection is. A ranking can use the
    distance itself, so it is computed here. A detection with no neighbour at
    all in its recording gets NaN, which a percentile rank places at the
    isolated end.
    """
    sc = _site_col(det_df, site_col)
    keys = [c for c in (sc, "recording", "species") if c in det_df.columns]
    start = pd.to_numeric(det_df["start_s"], errors="coerce")

    out = pd.Series(np.nan, index=det_df.index, dtype=float)
    grouper = det_df.groupby(keys, sort=False) if keys else [((), det_df)]
    for _, sub in grouper:
        s = start.loc[sub.index].sort_values(kind="stable")
        if len(s) < 2:
            continue
        v = s.to_numpy(dtype=float)
        prev = np.full(len(v), np.inf)
        nxt = np.full(len(v), np.inf)
        prev[1:] = v[1:] - v[:-1]
        nxt[:-1] = v[1:] - v[:-1]
        out.loc[s.index] = np.minimum(prev, nxt)
    return out


def available_candidates(det_df):
    """The subset of CANDIDATE_SIGNALS present and numeric in ``det_df``."""
    return {c: d for c, d in CANDIDATE_SIGNALS.items()
            if c in det_df.columns
            and pd.to_numeric(det_df[c], errors="coerce").notna().any()}
