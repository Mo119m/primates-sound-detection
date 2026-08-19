"""Bring the Drive copy of the packed index up to the expert's current verdicts.

The 3 GB image pack on Drive is still correct -- the same clips in the same
order -- but the two small CSVs beside it carry labels from before 2026-08-18,
and nine rows changed that day:

    2  IPA4ST clips  Background -> C_pogonias   (00778s, 00779s)
    1  IPA4ST clip   Background -> Cernic       (00812s)
    4  IPA2ST clips  Background -> C_pogonias   (from the auto_flagged_fp pass)
    1  IPA2ST clip   Background -> Cernic       (same pass)
    1  IPA2ST clip   dropped, "Potential C", species not established

Re-uploading three gigabytes to move nine labels is absurd, and the labels are
already in the repository this notebook clones, in two files of 29 and 89 rows
that carry no coordinates. So the fix is applied here, to the copies under
/content/dataC, after the Drive copy and before anything trains.

Rows are never added or removed. v13_features.npy is addressed by row number,
so deleting a row would silently shift every feature after it onto the wrong
clip; the one excluded clip is marked ok = False instead, which is the column
train_v13_loso.py already uses to skip a row while leaving the addressing
intact.

Both files are written because train_v13_loso.py proves the manifest and the
index are the same object before it trusts either, and disagreeing copies fail
that check rather than passing quietly.
"""
import os
import re
import sys

import pandas as pd

DATA = os.environ.get("DATAC", "/content/dataC")
REPO = os.environ.get("REPO", "/content/repo")
REL = os.path.join(REPO, "data/labels/relabel_2026-08-18.csv")
EXC = os.path.join(REPO, "data/labels/exclude_from_training_2026-08-17.csv")
TARGETS = ["v13_index.csv", "manifest.csv"]

# Recorder coordinates were removed from the label files, so the join has to
# ignore them on the manifest side too. Stripping is safe: it leaves all 6,496
# clip names distinct, the timestamp already being millisecond-resolution.
GPS = re.compile(r"[+-]\d{2}\.\d{4}[+-]\d{3}\.\d{4}")


def canon(name):
    return GPS.sub("", str(name))


def main():
    rel = pd.read_csv(REL)
    exc = pd.read_csv(EXC)
    new_label = dict(zip(rel["file"].map(canon), rel["new_label"]))
    new_source = dict(zip(rel["file"].map(canon), rel["new_source"]))
    excluded = set(exc["file"].map(canon))
    print(f"{len(rel)} relabels, {len(exc)} exclusions from the repository")

    both = set(new_label) & excluded
    if both:
        sys.exit(f"a clip is on both lists: {sorted(both)[:3]}")

    for name in TARGETS:
        p = os.path.join(DATA, name)
        if not os.path.exists(p):
            sys.exit(f"missing {p}; run the Drive copy cell first")
        d = pd.read_csv(p, keep_default_na=False)
        before = len(d)
        base = d["path"].map(lambda x: canon(os.path.basename(str(x))))

        hit = base.map(lambda b: b in new_label)
        for i in d.index[hit]:
            d.at[i, "label"] = new_label[base[i]]
            d.at[i, "source"] = new_source[base[i]]
            if "verified" in d.columns:
                d.at[i, "verified"] = True

        drop = base.map(lambda b: b in excluded)
        if "ok" in d.columns and drop.any():
            d.loc[drop, "ok"] = False

        assert len(d) == before, "row count must not change"
        d.to_csv(p, index=False)
        print(f"  {name:16s} relabelled {int(hit.sum())}, "
              f"marked not-ok {int(drop.sum())}, rows still {len(d)}")

    a = pd.read_csv(os.path.join(DATA, TARGETS[0]), keep_default_na=False)
    b = pd.read_csv(os.path.join(DATA, TARGETS[1]), keep_default_na=False)
    assert (a["label"].values == b["label"].values).all(), \
        "index and manifest disagree on label; training will refuse them"

    ok = (a["ok"].astype(str).str.lower().isin(("true", "1"))
          if "ok" in a.columns else pd.Series(True, index=a.index))
    print("\ntrainable rows:")
    for k, v in sorted(a[ok]["label"].value_counts().items()):
        print(f"  {k:18s} {v}")

    # The failure this project actually hit, twice: a target call training as a
    # negative. Cheap to assert, expensive to miss.
    neg = {"Background", "Colobus_confuser"}
    leaked = [base for base, lab in zip(
        a["path"].map(lambda x: canon(os.path.basename(str(x)))), a["label"])
        if base in new_label and lab in neg]
    assert not leaked, f"target call still a negative: {leaked[:3]}"
    print("\nno relabelled clip is still in the negative class")


if __name__ == "__main__":
    main()
