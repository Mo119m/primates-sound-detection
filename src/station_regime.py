"""
Decide, per station, which cleanup a station needs.

Field stations fail in two different ways, and the same filter cannot serve
both. Most stations accumulate scattered false positives, which respond to the
temporal-isolation rule. Occasionally a station is overrun by one species the
model was never trained on: those detections are numerous and acoustically
consistent, so they are neither isolated in time nor outliers in feature space,
and the ordinary filters cannot see them. Removing such a cluster wholesale
works there and is far too destructive anywhere else.

Telling the two apart has to be done before any labels exist, and without a
threshold that must hold across stations -- feature-space distances are not on
the same scale at every site, and every absolute cutoff tried against the
manual review failed to transfer. The rule here therefore compares a station
only against itself:

  1. order the station's detections by how tightly each clusters with its
     neighbours (the k-th nearest neighbour distance in feature space),
  2. split at the largest relative gap in that ordering,
  3. treat the tight side as an invading cluster only when it covers much of
     the station *and* sits further from the training data than that station's
     own remainder.

Step 3 is what distinguishes an intruding species from the target species
calling heavily. Both produce a dense cluster; the difference is that the
target's cluster is the familiar part of its station, while an intruder's is
the foreign part. Without it, the rule strips the stations where the target is
most abundant -- in the field review it discarded 121 genuine calls at the
highest-precision station of all.

This module holds no TensorFlow dependency so that both the cleanup pipeline
and the offline evaluation can use exactly the same rule.
"""
import numpy as np
import pandas as pd

# A station is only considered invaded when the dense group covers at least
# this share of its detections: a handful of similar calls is not an invasion.
DEFAULT_MIN_CLUSTER_FRACTION = 0.25
# ...and when the gap separating that group from the rest is at least this
# many times wide, so an evenly spread station is never split arbitrarily.
DEFAULT_MIN_GAP_RATIO = 2.0
# Neighbour rank used to measure tightness.
DEFAULT_K = 5

INVADED = "invaded"
NORMAL = "normal"


def knn_distance(feats, groups, k: int = DEFAULT_K):
    """
    Distance from each row to its ``k``-th nearest neighbour within its group.

    Small values mean the detection is one of many near-identical copies. A
    k-th neighbour distance is used rather than a count inside a fixed radius
    because any such radius would have to be chosen from the same distances it
    is meant to judge.
    """
    feats = np.asarray(feats, dtype=np.float32)
    groups = np.asarray(groups)
    out = np.full(len(feats), np.nan, dtype=np.float32)
    if len(feats) != len(groups):
        return out

    for g in pd.unique(groups):
        pos = np.flatnonzero(groups == g)
        if len(pos) <= k:
            continue
        f = feats[pos]
        d = np.linalg.norm(f[:, None, :] - f[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        out[pos] = np.partition(d, k - 1, axis=1)[:, k - 1]
    return out


def _split_at_largest_gap(values, min_frac, min_gap_ratio):
    """Index of the last member of the tight group, or None if no clear gap."""
    n = len(values)
    lo = max(1, int(min_frac * n))
    best_ratio, best_i = 0.0, None
    for i in range(lo - 1, n - 1):
        prev = max(values[i], 1e-9)
        ratio = values[i + 1] / prev
        if ratio > best_ratio:
            best_ratio, best_i = ratio, i
    if best_i is None or best_ratio < min_gap_ratio:
        return None
    return best_i


def detect_invading_cluster(df, tightness_col="recurrence_knn_dist",
                            distance_col="mahalanobis_d2",
                            group_col="station",
                            min_frac=DEFAULT_MIN_CLUSTER_FRACTION,
                            min_gap_ratio=DEFAULT_MIN_GAP_RATIO):
    """
    Mark detections belonging to a station's invading cluster.

    Returns a boolean Series aligned to ``df``. Stations without such a cluster
    contribute nothing, so the rule is a no-op wherever it has no business
    acting.
    """
    tight_vals = pd.to_numeric(df.get(tightness_col), errors="coerce")
    far_vals = pd.to_numeric(df.get(distance_col), errors="coerce")
    mask = pd.Series(False, index=df.index)
    if (tight_vals is None or far_vals is None or group_col not in df.columns
            or tight_vals.notna().sum() == 0 or far_vals.notna().sum() == 0):
        return mask

    for _, sub in df.groupby(group_col, sort=False):
        d = tight_vals.loc[sub.index].dropna()
        if len(d) < 10:
            continue
        order = d.sort_values()
        split = _split_at_largest_gap(order.to_numpy(), min_frac, min_gap_ratio)
        if split is None:
            continue

        cluster_idx = order.index[: split + 1]
        rest_idx = order.index[split + 1:]
        if len(cluster_idx) < min_frac * len(order) or not len(rest_idx):
            continue

        cluster_far = far_vals.loc[cluster_idx].dropna()
        rest_far = far_vals.loc[rest_idx].dropna()
        if not len(cluster_far) or not len(rest_far):
            continue
        # Foreign to this station, not merely dense within it.
        if cluster_far.median() > rest_far.median():
            mask.loc[cluster_idx] = True
    return mask


def classify_stations(df, mask=None, group_col="station", **kwargs):
    """Label each station ``invaded`` or ``normal`` from the cluster mask."""
    if mask is None:
        mask = detect_invading_cluster(df, group_col=group_col, **kwargs)
    if group_col not in df.columns:
        return pd.Series(NORMAL, index=df.index)
    invaded_sites = set(df.loc[mask, group_col].unique())
    return df[group_col].map(
        lambda s: INVADED if s in invaded_sites else NORMAL)
