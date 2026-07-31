"""
Tests for the grouped train/validation split.

The split these cover replaced a plain ``train_test_split`` over the *augmented*
images. That version reported 98.12 % validation accuracy next to 41.0 % field
precision, and the gap was the split: with an augmentation multiplier of 7 and a
20 % validation share, a clip keeps all its variants in training only 0.8**7 =
21 % of the time, so roughly four clips in five put a near-duplicate of a
validation image into the training set. Colobus was worse again -- its 617
windows come from 172 recordings at a 1 s hop, so neighbouring windows share half
their audio before augmentation.

These tests pin the two properties that fix: windows and augmentations of one
recording collapse to one group, and no group lands on both sides.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

train = pytest.importorskip("train")


# ---- source_group --------------------------------------------------

def test_overlapping_windows_share_a_group():
    """The 1 s-hop windows of one reference recording are one group."""
    windows = [f"colger100__t{t:03.1f}s.wav" for t in (0.0, 1.0, 2.0, 3.0)]
    assert {train.source_group(w) for w in windows} == {"colger100"}


def test_review_and_flagged_clips_of_one_recording_share_a_group():
    """Two naming conventions for the same recording must not split it."""
    a = "Cernic__20210222T053000+0100_Short-term_Makokou__01540s__conf0.980.wav"
    b = "Cernic__20210222T053000+0100_Short-term_Makokou__t00563s__conf0.86.wav"
    assert train.source_group(a) == train.source_group(b)


def test_birdnet_segments_of_one_recording_share_a_group():
    a = "0.554_2_20220218T023000+0100_Short-term_Makokou_807.0s_810.0s.wav"
    b = "0.730_1_20220218T023000+0100_Short-term_Makokou_285.0s_288.0s.wav"
    assert train.source_group(a) == train.source_group(b)


def test_distinct_recordings_stay_distinct():
    assert train.source_group("colger100__t000.0s.wav") != \
           train.source_group("colger101__t000.0s.wav")


# ---- build_source_groups -------------------------------------------

def test_augmented_variants_of_one_clip_share_a_group():
    sample_info = [f"Cernic_sample0_aug{j}" for j in range(7)]
    paths = {"Cernic": ["/x/colger100__t000.0s.wav"]}
    groups = train.build_source_groups(sample_info, paths, [])
    assert len(set(groups)) == 1


def test_background_samples_resolve_through_their_paths():
    sample_info = ["Background_sample0", "Background_sample1"]
    bg = [(np.zeros(4), "/x/bgA.wav"), (np.zeros(4), "/x/bgB.wav")]
    groups = train.build_source_groups(sample_info, {}, bg)
    assert groups[0] != groups[1]


def test_unresolvable_samples_get_their_own_group():
    """
    An origin that cannot be traced is isolated, never pooled.

    Pooling unknowns under one label would put unrelated clips in the same group;
    isolating them can only cost a little grouping power, and cannot create the
    leak this split exists to prevent.
    """
    groups = train.build_source_groups(["nonsense", "Cernic_sample9"],
                                       {"Cernic": []}, [])
    assert len(set(groups)) == 2


# ---- grouped_split -------------------------------------------------

def test_no_group_appears_on_both_sides():
    n_src, mult = 200, 7
    groups = np.array([f"s{i}" for i in range(n_src) for _ in range(mult)])
    y = np.array([i % 4 for i in range(n_src * mult)])
    X = np.arange(n_src * mult).reshape(-1, 1)

    X_tr, X_va, y_tr, y_va = train.grouped_split(X, y, groups, test_size=0.2)

    tr_groups = set(groups[X_tr.ravel()])
    va_groups = set(groups[X_va.ravel()])
    assert not (tr_groups & va_groups)
    assert len(X_va) > 0


def test_every_class_survives_into_validation():
    """A grouped split must not empty a small class out of validation."""
    groups, y = [], []
    for i in range(120):
        cls = 0 if i < 100 else (1 if i < 110 else 2)   # 100 / 10 / 10
        for _ in range(7):
            groups.append(f"s{i}")
            y.append(cls)
    groups, y = np.array(groups), np.array(y)
    X = np.arange(len(y)).reshape(-1, 1)

    _, _, _, y_va = train.grouped_split(X, y, groups, test_size=0.2)
    assert set(np.unique(y_va)) == {0, 1, 2}
