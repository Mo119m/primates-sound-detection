"""
Copy the files a second machine needs, and only those.

The working folder is 35 GB, but most of that either travels through git or
should be recomputed rather than carried:

- **24 GB is the feature cache.** It is a deterministic function of the image
  pack and the frozen VGG19 base. On a GPU it rebuilds in minutes; copying it
  over USB takes longer than regenerating it, and a stale copy that no longer
  matches the manifest is worse than no copy.
- **The source audio (5.8 GB)** is only needed to rebuild the image pack. The
  pack already contains every clip as the exact 224x224 image the model eats, so
  training and evaluation never touch it. It is in the optional tier for the
  case where the manifest changes -- after more clips are labelled, say -- and
  the pack has to be rebuilt.
- **Code, reviews and human labels** come from git.

What must travel is the pack, the manifest and index that index it, the review
table every evaluation is scored against, and the V12 weights that are the
baseline to beat.

Usage:
    python scripts/make_transfer_bundle.py --dest "/Volumes/Gabon CNN/v13_bundle"
    python scripts/make_transfer_bundle.py --dest <path> --full
    python scripts/make_transfer_bundle.py --dest <path> --dry-run
"""
import argparse
import os
import shutil
import sys

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

ESSENTIAL = [
    ("data/outputs/v13_images.npy",
     "the 31,678 clips as model-ready images -- the training input"),
    ("data/outputs/v13_index.csv",
     "row in the pack -> clip, label, station"),
    ("data/outputs/v13_manifest.csv",
     "the labelled dataset, with each clip's possible stations"),
    ("data/outputs/auto_cleanup/cleanup_vs_review.csv",
     "the 6,189 human verdicts -- every evaluation is scored against this"),
    ("data/outputs/models/best_model_v12.h5",
     "the baseline to beat"),
    ("data/outputs/detections",
     "V12's detections, for comparing a re-detection run"),
]

OPTIONAL = [
    ("data/outputs/detected_clips",
     "exported detection audio -- only to rebuild the image pack"),
    ("data/species", "reference clips -- only to rebuild"),
    ("data/background", "reference negatives -- only to rebuild"),
    ("data/outputs/auto_cleanup/auto_flagged_fp",
     "the auto-mined pool -- only to rebuild"),
]

SKIP = [
    ("data/outputs/v13_features.npy",
     "24 GB, and a deterministic function of the image pack. Regenerate it on "
     "the GPU\n     (train_v13_loso.py builds it automatically, minutes rather "
     "than an hour of copying)."),
]


def size_of(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def copy(rel, dest_root, dry_run):
    src = os.path.join(REPO, rel)
    dst = os.path.join(dest_root, rel)
    if not os.path.exists(src):
        print(f"  MISSING  {rel}")
        return 0
    n = size_of(src)
    print(f"  {n / 1e9:7.2f} GB  {rel}")
    if dry_run:
        return n
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
    return n


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", required=True,
                    help="Where to write the bundle (e.g. the external drive).")
    ap.add_argument("--full", action="store_true",
                    help="Also copy the source audio, so the other machine can "
                         "rebuild the image pack.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dest = os.path.abspath(args.dest)
    parent = os.path.dirname(dest)
    if not os.path.isdir(parent):
        sys.exit(f"destination's parent does not exist: {parent}")

    print(f"Bundle -> {dest}\n")
    print("ESSENTIAL")
    total = sum(copy(rel, dest, args.dry_run) for rel, _why in ESSENTIAL)

    if args.full:
        print("\nSOURCE AUDIO (only needed to rebuild the pack)")
        total += sum(copy(rel, dest, args.dry_run) for rel, _why in OPTIONAL)
    else:
        print("\nNOT COPIED (pass --full if the other machine must rebuild "
              "the pack)")
        for rel, why in OPTIONAL:
            print(f"  {size_of(os.path.join(REPO, rel)) / 1e9:7.2f} GB  "
                  f"{rel}\n     {why}")

    print("\nDELIBERATELY NOT COPIED")
    for rel, why in SKIP:
        print(f"  {rel}\n     {why}")

    print(f"\n{'Would copy' if args.dry_run else 'Copied'}: "
          f"{total / 1e9:.2f} GB")

    if not args.dry_run:
        with open(os.path.join(dest, "README.txt"), "w") as fh:
            fh.write(
                "V13 transfer bundle\n"
                "===================\n\n"
                "On the target machine:\n\n"
                "  git clone https://github.com/Mo119m/primates-sound-detection\n"
                "  cd primates-sound-detection\n"
                "  git checkout v13-honest-labels\n"
                "  cp -r <this folder>/data .\n"
                "  python scripts/check_gpu.py\n\n"
                "Then read HANDOFF.md -- it opens with a START HERE section.\n\n"
                "The feature cache (data/outputs/v13_features.npy, 24 GB) is\n"
                "deliberately absent. train_v13_loso.py rebuilds it from the\n"
                "image pack on first run; on a GPU that is minutes.\n")
        print(f"Wrote {os.path.join(dest, 'README.txt')}")


if __name__ == "__main__":
    main()
