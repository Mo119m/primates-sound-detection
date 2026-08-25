"""
Train V13 and score it leave-one-station-out against the human review.

What this measures, and what it cannot
--------------------------------------
The 6 189 reviewed detections are the only ground truth this project has that
was paid for in listening hours. Each is a 2 s window the deployed V12 fired on,
labelled by a person as a genuine *C. nictitans* call (2 535) or not (3 654).
Re-classifying those exact windows with a new model answers one question
exactly: **of the false positives a reviewer had to sit through, how many would
the new model no longer have produced, and how many real calls would it have
lost?** That is precision, and it is the number the field cares about.

It is not recall. A model that finds calls V12 missed cannot show that here,
because a window V12 never fired on was never exported and never reviewed. Recall
needs a re-detection pass over the recordings; those are 444 GB on an external
drive and about 170 h of CPU, so it is a separate job (see
``scripts/recall_sample.py`` for the sampling approach that bounds recall from a
few hours of exhaustively annotated audio).

Why leave-one-station-out
-------------------------
The obvious experiment -- fold the 3 654 reviewed false positives into training,
then check how many of them the model now rejects -- answers nothing, because it
tests on its own training data. Worse, the same trap is already in the *existing*
numbers: V12's training set contains auto-flagged negatives mined from the same
2021-02 recordings at IPA2, IPA10, IPA11, IPA13, IPA14 and IPA16, so its reported
41.0 % field precision is partly in-sample too.

So each fold holds out one station completely. A clip is withheld whenever the
held-out station appears in its ``possible_stations``, which includes the 1 348
clips whose filenames narrow the source to a group of stations without naming
one -- the five that recorded with GPS off write identical filenames, and
guessing between them would leak quietly and inflate every fold.

Usage:
    # one fold, to check the pipeline end to end
    python scripts/train_v13_loso.py --folds IPA20ST --epochs 8

    # verified-label baseline (a GPU job; ~113 h on this CPU)
    python scripts/train_v13_loso.py --verified-only --folds all \
        --out data/outputs/v13_loso_verified.csv
"""
import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import re
import socket
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TAP_LAYER = "block4_conv4"
DEFAULT_RESULTS_PATH = os.path.join(REPO, "data/outputs/v13_loso.csv")
IMAGE_SHAPE = (224, 224, 3)
FEATURE_CACHE_SHAPE = (28, 28, 512)
ARTIFACT_LOCK_KIND = "v13-artifact-lock"
ARTIFACT_LOCK_SCHEMA_VERSION = 1
MANIFEST_IDENTITY_COLUMNS = (
    "path", "label", "station", "possible_stations", "source", "verified",
    "period", "aug",
)


class ArtifactIntegrityError(RuntimeError):
    """The packed rows no longer describe the manifest the caller requested."""


def _require_columns(frame, columns, artifact):
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ArtifactIntegrityError(
            f"{artifact} is missing required column(s): {', '.join(missing)}")


def _text_values(values):
    """Compare CSV values without treating a blank cell and NaN differently."""
    return values.fillna("").astype(str)


def _bool_values(values, artifact, column):
    """Parse a CSV boolean strictly instead of letting ``'False'`` be truthy."""
    parsed, bad = [], []
    for row, value in values.items():
        if isinstance(value, (bool, np.bool_)):
            parsed.append(bool(value))
            continue
        if pd.isna(value):
            bad.append((row, value))
            parsed.append(False)
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            if value in (0, 1):
                parsed.append(bool(value))
                continue
        token = str(value).strip().casefold()
        if token in {"true", "t", "yes", "y", "1"}:
            parsed.append(True)
        elif token in {"false", "f", "no", "n", "0"}:
            parsed.append(False)
        else:
            bad.append((row, value))
            parsed.append(False)
    if bad:
        examples = ", ".join(f"row {int(row) + 1}: {value!r}"
                             for row, value in bad[:3])
        raise ArtifactIntegrityError(
            f"{artifact}.{column} must contain explicit true/false values "
            f"({examples})")
    return pd.Series(parsed, index=values.index, dtype=bool)


def _label_counts(frame):
    return {str(label): int(count)
            for label, count in frame["label"].value_counts().sort_index().items()}


def _format_counts(counts):
    return ", ".join(f"{label}={count}" for label, count in counts.items())


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_metadata(path, include_sha256=False):
    """Small, stable provenance for a file used by this run."""
    if not os.path.isfile(path):
        raise ArtifactIntegrityError(f"required file not found: {path}")
    stat = os.stat(path)
    metadata = {
        "path": os.path.abspath(path),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_sha256:
        metadata["sha256"] = _sha256(path)
    return metadata


def _npy_metadata(path, artifact, expected_rows=None, expected_trailing_shape=None,
                  expected_dtype=None, include_sha256=False):
    """Read an .npy header and enforce the complete artifact contract."""
    try:
        array = np.load(path, mmap_mode="r")
    except (OSError, ValueError) as error:
        raise ArtifactIntegrityError(
            f"could not read {artifact} {path}: {error}") from error
    try:
        rows = int(len(array))
        if expected_rows is not None and rows != expected_rows:
            raise ArtifactIntegrityError(
                f"{artifact} has {rows} rows but v13_index.csv has "
                f"{expected_rows}; rebuild the pack/index/cache as one unit")
        if (expected_trailing_shape is not None and
                tuple(array.shape[1:]) != tuple(expected_trailing_shape)):
            raise ArtifactIntegrityError(
                f"{artifact} has trailing shape {tuple(array.shape[1:])}, "
                f"expected {tuple(expected_trailing_shape)}")
        if expected_dtype is not None and np.dtype(array.dtype) != np.dtype(expected_dtype):
            raise ArtifactIntegrityError(
                f"{artifact} has dtype {array.dtype}, expected "
                f"{np.dtype(expected_dtype)}")
        metadata = _file_metadata(path, include_sha256=include_sha256)
        metadata.update({"rows": rows, "shape": list(array.shape),
                         "dtype": str(array.dtype)})
        return metadata
    finally:
        mmap = getattr(array, "_mmap", None)
        if mmap is not None:
            mmap.close()
        del array


def _validate_index_rows(index):
    _require_columns(index, ("row", "ok"), "v13_index.csv")
    rows = pd.to_numeric(index["row"], errors="coerce")
    if rows.isna().any() or not np.array_equal(rows.to_numpy(),
                                                np.floor(rows.to_numpy())):
        raise ArtifactIntegrityError(
            "v13_index.csv.row must be an integer address for every packed row")
    row_numbers = rows.astype(np.int64).to_numpy()
    expected = np.arange(len(index), dtype=np.int64)
    if not np.array_equal(row_numbers, expected):
        mismatch = int(np.flatnonzero(row_numbers != expected)[0])
        raise ArtifactIntegrityError(
            "v13_index.csv.row must be exactly 0..N-1 in pack order; "
            f"row {mismatch + 1} stores {row_numbers[mismatch]}, expected "
            f"{expected[mismatch]}")
    return _bool_values(index["ok"], "v13_index.csv", "ok")


def _validate_manifest_index(manifest, index):
    """Prove the manifest is the exact one used to create the image index."""
    _require_columns(manifest, MANIFEST_IDENTITY_COLUMNS, "v13_manifest.csv")
    _require_columns(index, MANIFEST_IDENTITY_COLUMNS, "v13_index.csv")
    if len(manifest) != len(index):
        raise ArtifactIntegrityError(
            f"manifest/index row-count mismatch: manifest has {len(manifest)}, "
            f"index has {len(index)}. Rebuild v13_images.npy, v13_index.csv, "
            "and v13_features.npy from this manifest together.")

    manifest_verified = _bool_values(manifest["verified"],
                                     "v13_manifest.csv", "verified")
    index_verified = _bool_values(index["verified"], "v13_index.csv",
                                  "verified")
    for column in MANIFEST_IDENTITY_COLUMNS:
        left = (manifest_verified if column == "verified"
                else _text_values(manifest[column]))
        right = (index_verified if column == "verified"
                 else _text_values(index[column]))
        different = left.to_numpy() != right.to_numpy()
        if different.any():
            row = int(np.flatnonzero(different)[0])
            raise ArtifactIntegrityError(
                f"manifest/index mismatch at packed row {row + 1}, column "
                f"{column!r}: manifest={manifest.iloc[row][column]!r}, "
                f"index={index.iloc[row][column]!r}. Repack before training.")
    return manifest_verified, index_verified


def _dataset_summary(index):
    verified = _bool_values(index["verified"], "selected index", "verified")
    augmentation = index["aug"].fillna(0)
    return {
        "rows": int(len(index)),
        "labels": _label_counts(index),
        "verified": {"true": int(verified.sum()),
                     "false": int((~verified).sum())},
        "augmentation": {
            "by_aug": {str(aug): int(count) for aug, count in
                       augmentation.value_counts().sort_index().items()},
            "nonzero_rows": int((pd.to_numeric(augmentation,
                                                 errors="coerce") > 0).sum()),
        },
        "sources": {str(source): int(count) for source, count in
                    index["source"].value_counts().sort_index().items()},
    }


def _write_run_metadata(path, metadata):
    """Atomically update a sidecar so an interrupted sweep remains auditable."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    partial = path + ".partial"
    with open(partial, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(partial, path)


def _write_csv_atomic(frame, path):
    """Replace a result table only after pandas has finished writing it."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    partial = path + ".partial"
    frame.to_csv(partial, index=False)
    os.replace(partial, path)


def _save_weights_atomic(head, path):
    """Keep an interrupted fold from looking like a usable weight file."""
    suffix = ".weights.h5"
    if not path.endswith(suffix):
        raise ArtifactIntegrityError(
            f"weight path must end in {suffix!r}: {path}")
    partial = path[:-len(suffix)] + ".partial" + suffix
    head.save_weights(partial)
    os.replace(partial, path)


def _refuse_interrupted_artifacts(*paths):
    """Fail closed instead of guessing whether another writer owns a partial."""
    partials = [f"{path}.partial" for path in paths
                if path and os.path.exists(f"{path}.partial")]
    if partials:
        raise ArtifactIntegrityError(
            "interrupted artifact(s) are present: " + ", ".join(partials) +
            ". Refusing to train or recover in place because another process "
            "may own them. Use a fresh, run-specific artifact path after "
            "inspecting the interrupted build.")


@contextlib.contextmanager
def _exclusive_cache_build_lock(cache_path):
    """Reserve a cache path without deleting a stale or possibly live partial."""
    lock_path = cache_path + ".build.lock"
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    try:
        descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as error:
        raise ArtifactIntegrityError(
            f"cache build lock already exists: {lock_path}. Refusing to guess "
            "whether another cache build is active; choose a fresh --cache path.") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "cache": os.path.abspath(cache_path),
            }, handle, sort_keys=True)
            handle.write("\n")
        yield lock_path
    finally:
        try:
            os.remove(lock_path)
        except OSError:
            pass


def _checker_module():
    """Load the checker by path so both CLI entry points share one contract."""
    checker_path = os.path.join(REPO, "scripts", "check_v13_artifacts.py")
    spec = importlib.util.spec_from_file_location("v13_artifact_checker", checker_path)
    if spec is None or spec.loader is None:
        raise ArtifactIntegrityError(f"could not load artifact checker: {checker_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_artifact_lock(lock_path, manifest_path, index_path, images_path,
                            cache_path):
    """Require a full-hash checker receipt before a reportable primary run."""
    try:
        with open(lock_path, "r", encoding="utf-8") as handle:
            lock = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactIntegrityError(
            f"could not read --artifact-lock {lock_path}: {error}") from error
    if (lock.get("kind") != ARTIFACT_LOCK_KIND or
            lock.get("schema_version") != ARTIFACT_LOCK_SCHEMA_VERSION):
        raise ArtifactIntegrityError(
            "--artifact-lock is not a supported immutable V13 artifact lock")

    fingerprints = lock.get("fingerprints")
    requirements = {
        "manifest": ("file_sha256", "canonical_rows_sha256"),
        "index": ("file_sha256", "canonical_rows_sha256"),
        "images": ("shape", "dtype", "file_bytes", "full_sha256"),
        "cache": ("shape", "dtype", "file_bytes", "full_sha256"),
    }
    missing = [f"{artifact}.{field}" for artifact, fields in requirements.items()
               for field in fields
               if not isinstance(fingerprints, dict) or
               not fingerprints.get(artifact, {}).get(field)]
    if missing:
        raise ArtifactIntegrityError(
            "--artifact-lock is not a full-hash receipt; missing " +
            ", ".join(missing) + ". Reissue it with "
            "check_v13_artifacts.py --full-hash --write-lock <new-path>.")

    checker = _checker_module()
    checker_args = checker.parse_args([
        "--manifest", os.path.abspath(manifest_path),
        "--index", os.path.abspath(index_path),
        "--images", os.path.abspath(images_path),
        "--cache", os.path.abspath(cache_path),
        "--lock", os.path.abspath(lock_path),
        "--full-hash",
    ])
    report = checker.run_checks(checker_args)
    if not report.get("ok"):
        errors = report.get("errors", [])
        summary = "; ".join(
            f"{item.get('code', 'error')}: {item.get('message', '')}"
            for item in errors[:3])
        raise ArtifactIntegrityError(
            "artifact lock validation failed" + (f" ({summary})" if summary else ""))
    return {
        "path": os.path.abspath(lock_path),
        "sha256": _sha256(lock_path),
        "checker_version": report.get("checker_version"),
        "full_hash_verified": True,
        "fingerprints": report.get("fingerprints", {}),
    }


def _guard_output_paths(out_path, head_dir, metadata_path, overwrite,
                        prepare_cache_only):
    """Do not turn a new experiment into an implicit resume/overwrite."""
    existing = []
    if os.path.exists(metadata_path) or os.path.exists(metadata_path + ".partial"):
        existing.append(f"run metadata: {metadata_path}")
    if not prepare_cache_only:
        if os.path.exists(out_path) or os.path.exists(out_path + ".partial"):
            existing.append(f"results CSV: {out_path}")
        if os.path.exists(head_dir):
            if not os.path.isdir(head_dir):
                raise ArtifactIntegrityError(
                    f"head output path exists but is not a directory: {head_dir}")
            existing.append(f"head directory: {head_dir}")
    if existing and not overwrite:
        raise ArtifactIntegrityError(
            "refusing to overwrite existing output(s): " + "; ".join(existing) +
            ". Use a fresh run directory, or pass --overwrite only to resume "
            "an intentionally replaceable run.")


def load_index(index_path, manifest_path=None, verified_only=False,
               return_metadata=False):
    """
    The rows to train on, each still pointing at its row in the image pack.

    ``row`` is the position in ``v13_images.npy``.  The manifest and index must
    be byte-for-byte equivalent in their row identity before filtering: using a
    newer manifest to relabel an older pack changes what every feature row means.
    Rebuild the pack/index/cache together when a manifest changes.
    """
    raw_index = pd.read_csv(index_path)
    _require_columns(raw_index, ("path", "label", "station",
                                 "possible_stations", "source", "verified"),
                     "v13_index.csv")
    index_ok = _validate_index_rows(raw_index)
    index_verified = _bool_values(raw_index["verified"], "v13_index.csv",
                                  "verified")

    manifest = None
    manifest_verified = None
    if manifest_path:
        if not os.path.exists(manifest_path):
            raise ArtifactIntegrityError(
                f"manifest not found: {manifest_path}. Refusing to train: "
                "the manifest is required to prove that labels and image rows "
                "belong to the same dataset.")
        manifest = pd.read_csv(manifest_path)
        manifest_verified, index_verified = _validate_manifest_index(
            manifest, raw_index)
    elif verified_only:
        raise ArtifactIntegrityError(
            "--verified-only requires --manifest so the verified status and "
            "labels can be checked against the packed index")
    else:
        print("  ! manifest validation bypassed by explicit --manifest ''; "
              "the resulting run is not reproducible against a manifest")

    renderable = raw_index.loc[index_ok].copy()
    renderable["verified"] = index_verified.loc[renderable.index].to_numpy()
    before_verified = _label_counts(renderable)
    if verified_only:
        # The statuses have just been compared row-for-row, so selecting with
        # the manifest mask filters both artifacts' labels before any fold is
        # built.  Do not accept an index-only ``verified`` flag here.
        selected = renderable.loc[
            manifest_verified.loc[renderable.index].to_numpy()].copy()
    else:
        selected = renderable
    selected["possible_stations"] = selected["possible_stations"].fillna("")
    selected["station"] = selected["station"].fillna("")
    selected = selected.reset_index(drop=True)

    selection = {
        "packed_rows": int(len(raw_index)),
        "renderable_rows": int(len(renderable)),
        "renderable_verified_rows": int(renderable["verified"].sum()),
        "renderable_unverified_rows": int((~renderable["verified"]).sum()),
        "labels_before_verified_filter": before_verified,
        "selected_rows": int(len(selected)),
        "labels_selected": _label_counts(selected),
        "verified_only": bool(verified_only),
    }
    print("Dataset selection:")
    print(f"  packed index rows: {selection['packed_rows']}")
    print(f"  renderable rows: {selection['renderable_rows']} "
          f"({selection['renderable_verified_rows']} verified, "
          f"{selection['renderable_unverified_rows']} unverified)")
    print("  labels before verified filter: " +
          _format_counts(selection["labels_before_verified_filter"]))
    if verified_only:
        print(f"  --verified-only selected {selection['selected_rows']} rows")
        print("  labels after verified filter: " +
              _format_counts(selection["labels_selected"]))

    metadata = {
        "manifest_index_validated": manifest is not None,
        "index": _file_metadata(index_path, include_sha256=True),
        "selection": selection,
    }
    if manifest is not None:
        metadata["manifest"] = _file_metadata(manifest_path, include_sha256=True)
    if return_metadata:
        return selected, metadata
    return selected


def feature_cache(images_path, cache_path, batch=64, replace_existing=False):
    """
    Run the frozen VGG19 once and keep the tap activations.

    The base is frozen for stage 1, so its output for a given image never
    changes between epochs or between folds. Recomputing it is the whole cost of
    training: on this CPU the base runs at 8.9 images/s and the head at 19, so a
    cached run is not an approximation of stage 1 -- it is stage 1, with the
    constant part evaluated once instead of sixteen times over.
    """
    _refuse_interrupted_artifacts(images_path, cache_path)
    images = np.load(images_path, mmap_mode="r")
    n = len(images)
    if os.path.exists(cache_path):
        try:
            metadata = _npy_metadata(
                cache_path, "feature cache", expected_rows=n,
                expected_trailing_shape=FEATURE_CACHE_SHAPE,
                expected_dtype=np.float16)
        except ArtifactIntegrityError as error:
            if not replace_existing:
                raise ArtifactIntegrityError(
                    f"existing feature cache is not reusable: {error}. "
                    "Use a fresh, run-specific --cache path, or pass "
                    "--replace-cache with --prepare-cache-only to atomically "
                    "replace this derived cache.") from error
            print(f"  replacing invalid cache {cache_path}: {error}")
        else:
            print(f"  reusing {cache_path} ({tuple(metadata['shape'])})")
            return np.load(cache_path, mmap_mode="r")

    # Build into a scratch name and rename only once every row is written.
    # The lock deliberately has no force-unlock mode: a stale lock or partial is
    # evidence that ownership is uncertain, so recovery must use a fresh path.
    with _exclusive_cache_build_lock(cache_path):
        _refuse_interrupted_artifacts(images_path, cache_path)
        if os.path.exists(cache_path):
            try:
                metadata = _npy_metadata(
                    cache_path, "feature cache", expected_rows=n,
                    expected_trailing_shape=FEATURE_CACHE_SHAPE,
                    expected_dtype=np.float16)
            except ArtifactIntegrityError:
                if not replace_existing:
                    raise
            else:
                print(f"  another process completed {cache_path}; reusing it")
                return np.load(cache_path, mmap_mode="r")

        import tensorflow as tf
        from tensorflow.keras.applications import VGG19

        base = VGG19(weights="imagenet", include_top=False,
                     input_shape=IMAGE_SHAPE)
        extractor = tf.keras.Model(base.input, base.get_layer(TAP_LAYER).output)
        shape = tuple(extractor.output.shape[1:])
        if shape != FEATURE_CACHE_SHAPE:
            raise ArtifactIntegrityError(
                f"{TAP_LAYER} produced {shape}, expected {FEATURE_CACHE_SHAPE}")
        print(f"  extracting {n} x {shape} -> {cache_path} "
              f"({n * np.prod(shape) * 2 / 1e9:.1f} GB float16)")

        partial = cache_path + ".partial"
        out = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float16,
                                        shape=(n,) + shape)
        t0 = time.time()
        for lo in range(0, n, batch):
            hi = min(n, lo + batch)
            x = images[lo:hi].astype("float32") / 255.0
            out[lo:hi] = extractor.predict(x, verbose=0).astype("float16")
            if lo % (batch * 20) == 0:
                rate = hi / max(time.time() - t0, 1e-9)
                print(f"\r  {hi}/{n}  {rate:.0f} img/s  "
                      f"eta {(n - hi) / rate / 60:.1f} min", end="", flush=True)
        out.flush()
        del out
        print()
        os.replace(partial, cache_path)
    return np.load(cache_path, mmap_mode="r")


def fold_masks(index, station, keep_all_background=False):
    """
    Training rows, the reviewed rows this fold is scored on, and the reviewed
    rows a deployable threshold may be fitted on.

    The third mask is the one that makes the headline number honest. A threshold
    read off the held-out station's own confirmed calls is an oracle: it needs
    the answer in order to be chosen, so the precision it buys cannot be
    reproduced by anyone running the method on a new site. Fitting it on the
    other fifteen stations instead is what deployment actually looks like, and
    the recall it then achieves on the held-out station is a result rather than
    a constraint.
    """
    possible = index["possible_stations"].str.split(";")
    withheld = possible.apply(lambda c: station in c)
    if keep_all_background:
        # Background is not withheld by station, because collecting it is part
        # of the method rather than a leak in the measurement. The pipeline's
        # third contribution is that a reviewer's rejected detections are
        # recycled as negatives; anyone deploying at a new site runs exactly
        # that loop and would have that site's rejected detections in training
        # before the second round. Withholding them measures a situation no
        # user is ever in.
        #
        # What this changes is what the number means, and the name has to
        # follow: with the station's own rejected detections in training, the
        # figure is the value of one round of review-and-retrain at a station,
        # not transfer to a station never heard. The station's confirmed CALLS
        # stay withheld -- those need an expert, a new site does not have them,
        # and they are the numerator of the very score being reported.
        withheld = withheld & (index["label"] != "Background")
    train = ~withheld
    reviewed = index["source"].str.startswith("review")
    evaluate = (index["station"] == station) & reviewed
    calibrate = reviewed & (index["station"] != station) & train
    return train.to_numpy(), evaluate.to_numpy(), calibrate.to_numpy()


def recall_threshold(scores, is_call, match_recall):
    """Lowest score that still keeps ``match_recall`` of the calls in a set."""
    cs = np.sort(np.asarray(scores)[np.asarray(is_call)])
    if not cs.size:
        return None
    k = int(np.floor((1 - match_recall) * cs.size))
    return float(cs[min(k, cs.size - 1)])


def _make_batches_class():
    """
    Build the batch reader on top of whichever base class Keras offers.

    It has to be a Keras dataset object, not a generator wrapped in
    ``tf.data.Dataset.from_generator``. Keras 3 consumes such a dataset **once**
    and then stops with "Your input ran out of data; interrupting training", so
    a run asked for 15 epochs silently trains for one. That failure prints a
    warning in the middle of normal output and otherwise looks like a completed
    run -- it produced smoke-test numbers here that read as a badly performing
    model rather than as a broken loop. ``PyDataset`` is re-iterated properly
    every epoch and gets ``on_epoch_end`` called.
    """
    import keras
    base = getattr(keras.utils, "PyDataset", None) or keras.utils.Sequence

    class MemmapBatches(base):
        """
        Batches read from a feature memmap, one at a time.

        The cache is 25 GB and this machine has 16 GB, so ``feats[rows]`` -- the
        obvious way to gather a fold's training set -- would materialise 22 GB
        and die. Reading per batch keeps the working set at tens of megabytes,
        which is what a memmap is for. Rows are sorted so an epoch walks the
        file forwards instead of seeking.
        """

        def __init__(self, feats, rows, labels, batch_size=32, shuffle=False,
                     scale=1.0, **kwargs):
            try:
                super().__init__(**kwargs)
            except TypeError:      # keras.utils.Sequence takes no kwargs
                super().__init__()
            self.feats = feats
            self.rows = np.sort(np.asarray(rows))
            self.labels = np.asarray(labels)
            self.batch_size = batch_size
            self.shuffle = shuffle
            # Feature caches are float16 activations and go through unchanged;
            # the image pack is uint8 0-255 and the base expects 0-1. Carrying
            # the factor here rather than at the call sites means the scoring
            # and calibration paths, which build their own batches, cannot be
            # left on the wrong scale.
            self.scale = float(scale)
            self._order = np.arange(len(self.rows))

        def __len__(self):
            return int(np.ceil(len(self.rows) / self.batch_size))

        def __getitem__(self, i):
            sel = np.sort(self._order[i * self.batch_size:
                                      (i + 1) * self.batch_size])
            r = self.rows[sel]
            x = np.asarray(self.feats[r], dtype="float32")
            if self.scale != 1.0:
                x = x * self.scale
            return x, self.labels[sel]

        def on_epoch_end(self):
            if self.shuffle:
                np.random.shuffle(self._order)

    return MemmapBatches


MemmapBatches = _make_batches_class()


def train_head(feats, rows, labels, groups, class_names, epochs, seed,
               verbose=0, pooling="temporal_freqpos", patience=3,
               unfreeze=0, finetune_epochs=5, finetune_lr=1e-5):
    import tensorflow as tf
    from sklearn.utils.class_weight import compute_class_weight
    import model as model_module
    import train as train_module

    local = np.arange(len(rows))
    tr, va, y_tr, y_va = train_module.grouped_split(
        local.reshape(-1, 1), labels, groups, test_size=0.2, seed=seed)
    tr, va = np.sort(tr.ravel()), np.sort(va.ravel())

    shape = feats.shape[1:]
    inp = tf.keras.Input(shape=shape)
    base = None
    if unfreeze:
        # Fine-tuning reads the image pack rather than the cached activations,
        # because the cache IS the frozen base's output and holding it fixed is
        # the thing being given up. The evidence for trying: in this model's
        # feature space the library training clips sit ~50 Mahalanobis units
        # from their class centre and field audio sits ~1860, and ImageNet
        # weights have no reason to place a close-range library recording of a
        # roar near a hundred-metre field recording of the same roar.
        from tensorflow.keras.applications import VGG19
        base = VGG19(weights="imagenet", include_top=False, input_tensor=inp)
        base.trainable = False
        tapped = base.get_layer(TAP_LAYER).output
        out = model_module.build_dense_tail(
            model_module.build_temporal_pool(tapped, pooling),
            num_classes=len(class_names))
    else:
        out = model_module.build_dense_tail(
            model_module.build_temporal_pool(inp, pooling),
            num_classes=len(class_names))
    head = tf.keras.Model(inp, out)
    head.compile(tf.keras.optimizers.Adam(1e-4),
                 "sparse_categorical_crossentropy", metrics=["accuracy"])

    present = np.unique(labels[tr])
    weights = compute_class_weight("balanced", classes=present, y=labels[tr])
    class_weight = {int(c): float(w) for c, w in zip(present, weights)}

    # On the record, because it is not visible anywhere else and it shapes every
    # number this script reports: 'balanced' hands each class an equal share of
    # the loss, so the two Colobus classes together carry half the gradient --
    # roughly 34x per clip against Background -- while `fold_masks` scores no
    # Colobus row in any of the 16 folds (every guereza clip is library audio
    # with no station). Half the training signal is spent on something this
    # evaluation cannot measure. Deliberately left alone here: the deployed
    # model is four-class, and changing the loss as well as the split and the
    # backbone would make a V13-vs-V12 difference unattributable. The two-class
    # ablation is the next run, not this one.
    print("   class weights: " + "  ".join(
        f"{class_names[int(c)]} {w:.2f}" for c, w in sorted(class_weight.items())))

    scale = 1.0 / 255.0 if unfreeze else 1.0
    train_seq = MemmapBatches(feats, rows[tr], labels[tr], shuffle=True,
                              scale=scale)
    val_seq = MemmapBatches(feats, rows[va], labels[va], scale=scale)

    # Early stopping on validation loss, which Sun et al. apply and this pipeline
    # did not: "Overfitting and generalization errors would appear if we trained
    # the model for too many epochs [...] we applied early stopping". Running a
    # fixed fifteen epochs on a task the head solves to 99 % in-sample spends the
    # later epochs pushing already-correct examples further from the boundary.
    # restore_best_weights matters more than the stopping itself -- without it
    # the weights kept are the last ones, not the best ones.
    callbacks = []
    if patience:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True,
            verbose=verbose))

    hist = head.fit(train_seq, validation_data=val_seq, epochs=epochs,
                    class_weight=class_weight, callbacks=callbacks,
                    verbose=verbose)
    ran = len(hist.history.get("loss", []))
    # A short run means one of two very different things. Early stopping firing
    # is the intended behaviour; the input pipeline being consumed once instead
    # of per epoch is the bug this guard was written for, and it produced runs
    # that looked like a weak model rather than a broken loop. The callback's
    # stopped_epoch separates them: it is 0 when early stopping never fired.
    stopped_early = any(getattr(c, "stopped_epoch", 0) for c in callbacks)
    if ran < epochs and not stopped_early:
        raise RuntimeError(
            f"training stopped after {ran} of {epochs} epochs without early "
            f"stopping firing -- the input pipeline is being consumed once "
            f"instead of per epoch, and the resulting numbers would look like "
            f"a weak model rather than a broken loop")
    if stopped_early:
        best = int(np.argmin(hist.history["val_loss"])) + 1
        print(f"   early stop after {ran} epochs, best val_loss at epoch {best}")

    if unfreeze and base is not None:
        # Two stages, the order that matters: a randomly initialised head
        # backpropagating into pretrained convolutions destroys them, so the
        # head is fitted first above and only then are the last blocks released,
        # at a learning rate two orders of magnitude below the head's.
        # Count blocks back from the tap, not from block5. The model taps
        # block4_conv4, so block4_pool and the whole of block5 are downstream of
        # the output and are not in the graph at all. Counting from block5 meant
        # --unfreeze 1 released only block5 and trained nothing, while
        # --unfreeze 2 reported "unfreezing ['block5', 'block4'] (10 layers)"
        # when four layers and 8.26M parameters actually received gradients.
        # Neither is visible in the loss curve, because a run that fine-tunes
        # nothing looks exactly like a run whose fine-tuning did not help.
        tap_block = int(TAP_LAYER[len("block"):].split("_")[0])
        blocks = [f"block{tap_block - i}" for i in range(unfreeze)]

        in_graph = {l.name for l in head.layers}
        released, inert = [], []
        for layer in base.layers:
            hit = any(layer.name.startswith(b) for b in blocks)
            layer.trainable = hit
            if hit:
                (released if layer.name in in_graph else inert).append(layer.name)
        base.trainable = True
        for layer in base.layers:
            if not any(layer.name.startswith(b) for b in blocks):
                layer.trainable = False
        head.compile(tf.keras.optimizers.Adam(finetune_lr),
                     "sparse_categorical_crossentropy", metrics=["accuracy"])
        n_par = sum(int(np.prod(w.shape)) for w in head.trainable_weights)
        print(f"   unfreezing {blocks}: {len(released)} layers in the graph, "
              f"{n_par/1e6:.2f}M trainable params, lr {finetune_lr:g}, "
              f"{finetune_epochs} epochs")
        if inert:
            print(f"   ! {len(inert)} requested layers are downstream of the "
                  f"{TAP_LAYER} tap and get no gradient: {', '.join(inert)}")
        if not released:
            print(f"   ! nothing was released -- this run is not fine-tuned. "
                  f"With the tap at {TAP_LAYER}, --unfreeze must be >= 1 "
                  f"counting back from block{tap_block}.")
        ft_cb = []
        if patience:
            ft_cb.append(tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience,
                restore_best_weights=True, verbose=verbose))
        head.fit(train_seq, validation_data=val_seq, epochs=finetune_epochs,
                 class_weight=class_weight, callbacks=ft_cb, verbose=verbose)
    val_acc = head.evaluate(val_seq, verbose=0)[1]
    # `va` is returned so the caller can calibrate a threshold on rows the head
    # never took a gradient step on. Calibrating on the rest of the training
    # pool, which is what this script did until an audit measured it, fits the
    # threshold to memorised scores: those sit higher than scores on fresh
    # audio, so the threshold comes out too high, and applying it to the
    # held-out station keeps fewer calls and reports a better precision than the
    # method would achieve in use. The per-fold recall spread of 0.66 to 0.99
    # was that miscalibration showing.
    return head, float(val_acc), rows[va]


_GPS_IN_NAME = re.compile(r"[+-]\d{2}\.\d{4}[+-]\d{3}\.\d{4}")


def _canon_review_key(name):
    """A filename with the recorder's coordinates removed, for joining on."""
    return _GPS_IN_NAME.sub("", str(name))


def detection_hours(index, review=None):
    """
    Wall-clock hour of each reviewed detection, or NaN where it is not known.

    Taken from the review table's ``timestamp`` + ``start_s`` columns rather
    than parsed out of the filename. Two filename schemes coexist in this
    corpus -- ``S<start>+0100_E<end>+0100_<lat><lon>`` and
    ``<start>+0100_Short-term_Makokou`` -- and a regex written against one of
    them silently drops the other. That is not hypothetical: it is how the
    Colobus "90.7 % of detections fall at night" figure was produced from 97 of
    253 files when the true value over all 253 is 48.6 %. Joining on basename
    resolves 6 189 of 6 189 reviewed rows and cannot half-match.
    """
    hours = pd.Series(np.nan, index=index.index)
    if review is None:
        # The full table first, the coordinate-free subset second. Only the
        # subset can go on Drive: cleanup_vs_review.csv carries the recorder's
        # position in 2,882 of its filenames, and a run without either writes a
        # CSV silently missing every gated_ column -- which is what three Colab
        # folds did on 2026-08-24 before anyone read their headers.
        for cand in ("data/outputs/auto_cleanup/cleanup_vs_review.csv",
                     "data/outputs/auto_cleanup/review_gate_table.csv"):
            p = os.path.join(REPO, cand)
            if os.path.exists(p):
                review = p
                break
    if review is None or not os.path.exists(review):
        print("  no review table found -- gated columns unavailable. Build the "
              "shareable one with scripts/make_review_gate_table.py")
        return hours
    rev = pd.read_csv(review)
    det = (pd.to_datetime(rev["timestamp"], format="%Y%m%dT%H%M%S")
           + pd.to_timedelta(rev["start_s"], unit="s"))
    # Canonicalise both sides. The shareable table has the coordinates stripped
    # out of its keys and the full one does not, so joining on the raw name
    # would match every row against one table and none against the other --
    # and "none" here does not raise, it just empties the gated columns.
    # Stripping is lossless: all 6,189 names stay distinct and the same 6,478
    # index rows match either way. Measured, not assumed.
    by_file = dict(zip(rev["file"].map(_canon_review_key),
                       det.dt.hour + det.dt.minute / 60.0))
    base = index["path"].map(lambda p: _canon_review_key(os.path.basename(str(p))))
    hours = base.map(by_file)
    matched = int(hours.notna().sum())
    print(f"  detection times: {matched}/{len(rev)} reviewed rows matched"
          f"  [{os.path.basename(review)}]")
    if matched == 0:
        print("  ! nothing matched. The gated_ columns will be empty and every "
              "comparison that quotes them will be unusable.")
    return hours


def fires_as_cernic(probs, class_names):
    """
    Reproduce the DEPLOYED decision rule, which argmaxes over detection groups.

    ``src/detection.py:236-241`` collapses the softmax onto
    ``config.DETECTION_GROUPS`` before taking the argmax, which routes
    ``Colobus_confuser``'s mass into Background. Taking ``probs.argmax()`` over
    the raw classes instead is a strictly weaker test: Cernic only has to beat
    each class individually rather than the summed Background group. It
    therefore fires on windows the pipeline would reject, and every figure
    derived from it -- ``kept_calls``, ``fps_removed``, ``v13_precision``,
    ``review_reduction`` -- is biased in one direction, too generous.

    The bias is not hypothetical: the evaluation population is V12's own false
    positives, which is exactly the subset engineered to carry high confuser
    mass.
    """
    import config
    groups = {}
    for i, name in enumerate(class_names):
        groups.setdefault(config.DETECTION_GROUPS.get(name, name), []).append(i)
    names = list(groups)
    stacked = np.stack([probs[:, groups[g]].sum(axis=1) for g in names], axis=1)
    return np.asarray(names)[stacked.argmax(axis=1)] == "Cernic"


def score_fold(head, feats, rows, truth, class_names, threshold,
               match_recall=0.95, hours=None, gate=None,
               cal_rows=None, cal_truth=None, cal_hours=None):
    """
    Re-classify one station's reviewed windows and compare with the review.

    Scored three ways, because one operating point is not a comparison. A model
    that fires less often removes false positives and loses calls together, so
    "68 % of false positives removed" means nothing without the recall it was
    bought at -- the degenerate model that predicts Background for everything
    removes 100 % of them.

    - At the deployment threshold: what the pipeline would actually do today.
    - At **matched recall**: the threshold that keeps ``match_recall`` of this
      station's confirmed calls. This compares the two models' *rankings* at
      equal recall, which is the right question when asking whether V13 orders
      calls above false positives better than V12 does. It is an upper bound and
      not an operating point: the threshold is read off the held-out station's
      own labels, so reproducing it requires already knowing the answer. Across
      the sixteen folds it ranges from 0.004 to 0.9964, which is what an oracle
      looks like from the outside.
    - At the **LOSO-fitted threshold**: the same rule fitted on the other
      fifteen stations and applied here, under the deployed grouped-argmax. This
      is the number a reader can reproduce, and the recall it achieves is
      measured rather than imposed. Expect it to be lower than the matched one;
      the gap between them is the cost of not knowing the answer in advance.

    The full sweep is returned too, so the trade-off can be read rather than
    argued about.
    """
    order = np.argsort(rows)
    truth = np.asarray(truth)[order]
    if hours is not None:
        hours = np.asarray(hours)[order]
    seq = MemmapBatches(feats, rows, np.zeros(len(rows)), batch_size=64,
                        scale=(1.0 / 255.0 if feats.dtype == np.uint8 else 1.0))
    probs = np.concatenate([head.predict(seq[i][0], verbose=0)
                            for i in range(len(seq))])
    cernic = class_names.index("Cernic")
    score = probs[:, cernic]
    top_is_cernic = fires_as_cernic(probs, class_names)

    is_call = truth == "call"
    n_call, n_fp = int(is_call.sum()), int((~is_call).sum())

    def at(t, require_top=True, mask=None):
        fires = (score >= t) & (top_is_cernic if require_top else True)
        if mask is not None:
            fires = fires & mask
        kc, kf = int((fires & is_call).sum()), int((fires & ~is_call).sum())
        return kc, kf

    kc, kf = at(threshold)
    kept = kc + kf
    row = {
        "detections": n_call + n_fp,
        "calls": n_call,
        "false_positives": n_fp,
        "v12_precision": round(n_call / max(n_call + n_fp, 1), 4),
        "kept_calls": kc,
        "kept_false_positives": kf,
        "calls_retained": round(kc / n_call, 4) if n_call else None,
        "fps_removed": round(1 - kf / n_fp, 4) if n_fp else None,
        "v13_precision": round(kc / kept, 4) if kept else None,
        "review_reduction": round(1 - kept / max(n_call + n_fp, 1), 4),
    }

    # Matched recall: the lowest score that still keeps `match_recall` of the
    # calls. Ignore the argmax rule here -- at a low threshold the question is
    # how the Cernic score ranks calls against false positives, not which class
    # happens to win.
    if n_call:
        t_match = recall_threshold(score, is_call, match_recall)
        kc_m, kf_m = at(t_match, require_top=False)
        row.update({
            "matched_threshold": round(t_match, 4),
            "matched_calls_retained": round(kc_m / n_call, 4),
            "matched_fps_removed": round(1 - kf_m / n_fp, 4) if n_fp else None,
            "matched_precision": (round(kc_m / (kc_m + kf_m), 4)
                                  if kc_m + kf_m else None),
        })

    # The deployable operating point. Everything above this line either uses a
    # fixed threshold or one read off the answers; this block fits the threshold
    # where a practitioner could fit it, on the stations that were in training,
    # and then reports what it does here. require_top stays on, because the
    # deployed pipeline collapses the softmax onto DETECTION_GROUPS before the
    # argmax and a comparison that quietly drops that rule flatters the model
    # (see fires_as_cernic).
    if n_call and cal_rows is not None and len(cal_rows):
        cal_order = np.argsort(cal_rows)
        cal_seq = MemmapBatches(
            feats, np.asarray(cal_rows)[cal_order], np.zeros(len(cal_rows)),
            batch_size=64,
            scale=(1.0 / 255.0 if feats.dtype == np.uint8 else 1.0))
        cal_probs = np.concatenate([head.predict(cal_seq[i][0], verbose=0)
                                    for i in range(len(cal_seq))])
        cal_score = cal_probs[:, cernic]
        cal_is_call = np.asarray(cal_truth)[cal_order] == "call"
        row["calibration_windows"] = int(len(cal_rows))

        t_loso = recall_threshold(cal_score, cal_is_call, match_recall)
        if t_loso is not None:
            kc_l, kf_l = at(t_loso)
            row.update({
                "loso_threshold": round(t_loso, 4),
                "loso_calls_retained": round(kc_l / n_call, 4),
                "loso_fps_removed": (round(1 - kf_l / n_fp, 4)
                                     if n_fp else None),
                "loso_precision": (round(kc_l / (kc_l + kf_l), 4)
                                   if kc_l + kf_l else None),
            })

        if cal_hours is not None and hours is not None and gate is not None:
            lo_g, hi_g = gate
            cal_in = np.asarray(cal_hours)[cal_order]
            cal_in = (cal_in >= lo_g) & (cal_in < hi_g)
            t_gl = recall_threshold(cal_score[cal_in], cal_is_call[cal_in],
                                    match_recall)
            g_in = (hours >= lo_g) & (hours < hi_g)
            gc, gf = int((is_call & g_in).sum()), int((~is_call & g_in).sum())
            if t_gl is not None and gc:
                kc_g, kf_g = at(t_gl, mask=g_in)
                row.update({
                    "gated_loso_threshold": round(t_gl, 4),
                    "gated_loso_calls_retained": round(kc_g / gc, 4),
                    "gated_loso_fps_removed": (round(1 - kf_g / gf, 4)
                                               if gf else None),
                    "gated_loso_precision": (round(kc_g / (kc_g + kf_g), 4)
                                             if kc_g + kf_g else None),
                })

    # The same comparison inside the deployed time window. Without this the
    # sweep measures V13 against V12 on a population that includes the
    # nocturnal detections the gate now discards for free -- 2 618 of the 6 189
    # reviewed windows, 2 586 of them false positives. V13 would be credited
    # with rejecting an insect chorus that costs nothing to reject, and the
    # marginal value of the model, which is what the paper is about, would be
    # hidden. These columns are the honest ones once TIME_FILTER is on.
    if hours is not None and gate is not None:
        lo, hi = gate
        inside = (hours >= lo) & (hours < hi)
        g_call = int((is_call & inside).sum())
        g_fp = int((~is_call & inside).sum())
        row["gated_detections"] = g_call + g_fp
        row["gated_v12_precision"] = (round(g_call / (g_call + g_fp), 4)
                                      if g_call + g_fp else None)
        if g_call and g_fp:
            gs = np.sort(score[is_call & inside])
            k = int(np.floor((1 - match_recall) * g_call))
            t_g = float(gs[min(k, g_call - 1)])
            kc_g, kf_g = at(t_g, require_top=False, mask=inside)
            row.update({
                "gated_matched_threshold": round(t_g, 4),
                "gated_matched_calls_retained": round(kc_g / g_call, 4),
                "gated_matched_fps_removed": round(1 - kf_g / g_fp, 4),
                "gated_matched_precision": (round(kc_g / (kc_g + kf_g), 4)
                                            if kc_g + kf_g else None),
            })
    return row


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default=os.path.join(REPO, "data/outputs/v13_images.npy"))
    ap.add_argument("--index", default=os.path.join(REPO, "data/outputs/v13_index.csv"))
    ap.add_argument("--cache", default=os.path.join(REPO, "data/outputs/v13_features.npy"))
    ap.add_argument("--manifest",
                    default=os.path.join(REPO, "data/outputs/v13_manifest.csv"),
                    help="The manifest used to create the image pack. It must "
                         "match v13_index.csv exactly in row order, labels, and "
                         "verified status. Pass '' only to explicitly bypass "
                         "manifest validation for a legacy non-verified run.")
    ap.add_argument("--verified-only", action="store_true",
                    help="Train only rows whose manifest and index both mark "
                         "verified=True. Requires a validated --manifest and "
                         "filters before station folds are built. A reportable "
                         "run also requires --artifact-lock.")
    ap.add_argument("--artifact-lock", default=None,
                    help="Immutable full-hash receipt written by "
                         "check_v13_artifacts.py --full-hash --write-lock. "
                         "Required for --verified-only training.")
    ap.add_argument("--out", default=DEFAULT_RESULTS_PATH)
    ap.add_argument("--head-dir", default=None,
                    help="Where to write per-fold head weights. Default: a "
                         "directory derived from --out (for example "
                         "v13_loso_verified_heads), never the shared "
                         "v13_heads directory.")
    ap.add_argument("--run-metadata", default=None,
                    help="JSON sidecar recording arguments, exact manifest/index "
                         "hashes, selected-label counts, and output paths. "
                         "Default: <out stem>.run.json.")
    ap.add_argument("--prepare-cache-only", action="store_true",
                    help="Validate the manifest/index/image-pack contract, "
                         "build or validate a feature cache, record metadata, "
                         "then exit before training any head. Use this to repair "
                         "a stale cache; it never writes fold heads or results. "
                         "It refuses a cache .partial; use a fresh run-specific "
                         "--cache path rather than deleting an interrupted build.")
    ap.add_argument("--replace-cache", action="store_true",
                    help="Only with --prepare-cache-only: atomically replace an "
                         "existing invalid cache after building a complete new "
                         "one. Never bypasses a .partial or build lock.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Allow an intentional restart to replace existing result, "
                         "head, or metadata outputs. This is not a resume; new "
                         "runs refuse by default.")
    ap.add_argument("--keep-all-background", action="store_true",
                    help="Do not withhold Background by station. Collecting "
                         "negatives from a site's own rejected detections is "
                         "part of this method, not a leak, so a fold that "
                         "withholds them measures a situation no user is in. "
                         "The station's confirmed calls stay withheld. Note "
                         "that the resulting figure is the value of one round "
                         "of review-and-retrain at a station, NOT transfer to "
                         "an unheard station, and must not be reported as a "
                         "held-out precision.")
    ap.add_argument("--folds", default="IPA20ST",
                    help="'all', or a comma-separated list of stations.")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drop-colobus", action="store_true",
                    help="Ablation: drop the Colobus_guereza class and fold "
                         "Colobus_confuser into Background. The two Colobus "
                         "classes take half the balanced loss (one guereza clip "
                         "carries ~34x the gradient of one Background clip) "
                         "while fold_masks scores zero Colobus rows in any of "
                         "the 16 folds -- every guereza clip is library audio "
                         "with no station. This measures what that costs "
                         "Cernic. Confuser clips are kept as Background rather "
                         "than discarded: they are 654 real field negatives, "
                         "and deployment already routes them there "
                         "(config.DETECTION_GROUPS).")
    ap.add_argument("--drop-pogonias", action="store_true",
                    help="Ablation: fold the C_pogonias class into Background. "
                         "The pogonias clips carry no station in their filenames "
                         "and fingerprint matching found no match against any "
                         "clip this repository has exported, so "
                         "possible_stations() returns an empty list and they "
                         "train in every fold including the one that holds their "
                         "station out -- if they came from the deployment at all. "
                         "Running the sweep with and without the class is what "
                         "separates the gain the class actually buys from the "
                         "gain unattributable audio could be leaking. Folded into "
                         "Background rather than discarded because deployment "
                         "already routes them there "
                         "(config.DETECTION_GROUPS).")
    ap.add_argument("--drop-extra-confuser", action="store_true",
                    help="Ablation: drop everything sourced colobus_field_fp "
                         "from Colobus_confuser, keeping only the reference "
                         "material. The counts moved with the dataset and this "
                         "text used to name them -- 253 detections against an "
                         "original 654 -- so it names the source instead, which "
                         "is what the code matches on. In the 2026-08-17 build "
                         "that is 307 clips of 961, or 1,038 augmented rows of "
                         "3,000. Those clips are all Colobus-channel false "
                         "positives, so adding them pushes the model to be more "
                         "conservative "
                         "about Colobus, and the retrained heads do respond "
                         "less to the field positive control than the deployed "
                         "model does. That could be this class or it could be "
                         "the frozen trunk; running with and without is what "
                         "separates them.")
    ap.add_argument("--pooling", default="temporal_freqpos",
                    choices=["temporal", "temporal_freq", "temporal_freqpos"],
                    help="Which head to train, for the architecture ablation "
                         "the paper currently describes but does not quantify. "
                         "'temporal' pools frequency away entirely, the V7 "
                         "path; 'temporal_freq' splits four bands with their "
                         "own Conv1D streams; 'temporal_freqpos' adds the "
                         "CoordConv frequency channel on top and is the "
                         "deployed head. Everything is downstream of the "
                         "cached features, so a comparison costs only the head "
                         "training.")
    ap.add_argument("--unfreeze", type=int, default=0, metavar="N",
                    help="Release the last N VGG19 blocks and fine-tune them "
                         "after the head is fitted. 0 keeps the base frozen, "
                         "which is what every result so far uses and what Sun "
                         "et al. do. Reads the image pack instead of the "
                         "feature cache, because the cache is the frozen "
                         "base's output: on a CPU this is roughly eight times "
                         "slower per epoch, so it is a GPU option.")
    ap.add_argument("--finetune-epochs", type=int, default=5)
    ap.add_argument("--finetune-lr", type=float, default=1e-5,
                    help="Two orders below the head's 1e-4. Pretrained "
                         "convolutions are destroyed by large updates, which "
                         "is the failure mode fine-tuning is known for.")
    ap.add_argument("--patience", type=int, default=3,
                    help="Early-stopping patience on validation loss, with the "
                         "best weights restored. Sun et al. stop early for the "
                         "reason that applies here: the head reaches 99 %% "
                         "in-sample, so later epochs only push correct examples "
                         "further from the boundary. 0 disables it and "
                         "reproduces the fixed-epoch runs.")
    ap.add_argument("--verbose", type=int, default=0)
    ap.add_argument("--max-train", type=int, default=None,
                    help="Cap the training pool per fold. For checking that the "
                         "pipeline runs on a CPU; a capped run's numbers are "
                         "not the experiment.")
    args = ap.parse_args()

    if args.replace_cache and not args.prepare_cache_only:
        sys.exit("--replace-cache is only valid with --prepare-cache-only")
    if args.verified_only and args.prepare_cache_only:
        sys.exit(
            "--verified-only is a training selection, not a cache build mode. "
            "Prepare the full cache first, issue its artifact lock, then start "
            "the verified-only run.")
    if args.verified_only and not args.prepare_cache_only and not args.artifact_lock:
        sys.exit(
            "--verified-only requires --artifact-lock from a passing full-hash "
            "check_v13_artifacts.py run before training")
    if args.prepare_cache_only and args.artifact_lock:
        sys.exit(
            "--artifact-lock is checked before training, after a cache exists. "
            "Do not pass it with --prepare-cache-only.")

    # The verified-only run is the reportable primary experiment, not a smoke
    # test.  Reusing the historic generic CSV path would overwrite an earlier
    # result before the new run has even passed the artifact gate.  Require an
    # explicit, run-specific output path while leaving legacy invocations
    # unchanged.
    if (args.verified_only and
            os.path.normcase(os.path.abspath(args.out)) ==
            os.path.normcase(os.path.abspath(DEFAULT_RESULTS_PATH))):
        sys.exit(
            "--verified-only requires an explicit --out path in a new run "
            "directory (for example data/outputs/v13_runs/<run_id>/results.csv); "
            "refusing to overwrite the historic v13_loso.csv")

    out_path = os.path.abspath(args.out)
    out_stem, _ = os.path.splitext(out_path)
    head_dir = (os.path.abspath(args.head_dir) if args.head_dir
                else out_stem + "_heads")
    metadata_path = (os.path.abspath(args.run_metadata) if args.run_metadata
                     else out_stem + ".run.json")
    if os.path.normcase(metadata_path) == os.path.normcase(out_path):
        sys.exit("--run-metadata must not be the same file as --out")
    if os.path.normcase(head_dir) in {
            os.path.normcase(out_path), os.path.normcase(metadata_path)}:
        sys.exit("--head-dir must be a distinct directory, not --out or --run-metadata")
    try:
        _guard_output_paths(out_path, head_dir, metadata_path, args.overwrite,
                            args.prepare_cache_only)
    except ArtifactIntegrityError as error:
        sys.exit(f"Refusing to start run: {error}")

    artifact_lock_metadata = None
    if args.artifact_lock:
        try:
            artifact_lock_metadata = _validate_artifact_lock(
                args.artifact_lock, args.manifest, args.index, args.images,
                args.cache)
        except ArtifactIntegrityError as error:
            sys.exit(f"Refusing to train: {error}")

    import config
    import train as train_module

    try:
        _refuse_interrupted_artifacts(args.index, args.images, args.cache)
        index, input_metadata = load_index(
            args.index, args.manifest, verified_only=args.verified_only,
            return_metadata=True)
        if artifact_lock_metadata is not None:
            input_metadata["artifact_lock"] = artifact_lock_metadata
        packed_rows = input_metadata["selection"]["packed_rows"]
        input_metadata["image_pack"] = _npy_metadata(
            args.images, "image pack", expected_rows=packed_rows,
            expected_trailing_shape=IMAGE_SHAPE, expected_dtype=np.uint8)
        cache_ready = False
        if os.path.exists(args.cache):
            try:
                cache_metadata = _npy_metadata(
                    args.cache, "feature cache", expected_rows=packed_rows,
                    expected_trailing_shape=FEATURE_CACHE_SHAPE,
                    expected_dtype=np.float16)
            except ArtifactIntegrityError as error:
                input_metadata["feature_cache"] = {
                    "path": os.path.abspath(args.cache),
                    "present": True,
                    "reusable": False,
                    "validation_error": str(error),
                }
                if not args.prepare_cache_only:
                    raise ArtifactIntegrityError(
                        "feature cache is not reusable; run "
                        "--prepare-cache-only with a fresh --cache path before "
                        "training") from error
                if not args.replace_cache:
                    raise ArtifactIntegrityError(
                        "feature cache is not reusable; use a fresh, "
                        "run-specific --cache path, or explicitly pass "
                        "--replace-cache with --prepare-cache-only") from error
            else:
                cache_metadata["reusable"] = True
                input_metadata["feature_cache"] = cache_metadata
                cache_ready = True
        else:
            input_metadata["feature_cache"] = {
                "path": os.path.abspath(args.cache), "present": False,
                "reusable": False,
            }
    except ArtifactIntegrityError as error:
        sys.exit(f"Refusing to train: {error}")

    if not cache_ready and not args.prepare_cache_only:
        sys.exit(
            "Refusing to train: no validated feature cache is available. Run "
            "--prepare-cache-only with a fresh, run-specific --cache path, then "
            "issue a full-hash artifact lock.")

    if index.empty:
        sys.exit("Refusing to train: no renderable rows remain after the "
                 "requested dataset filters")
    if args.drop_colobus:
        before = len(index)
        index = index[index["label"] != "Colobus_guereza"].copy()
        n_conf = int((index["label"] == "Colobus_confuser").sum())
        index.loc[index["label"] == "Colobus_confuser", "label"] = "Background"
        index = index.reset_index(drop=True)
        print(f"Ablation --drop-colobus: removed {before - len(index)} guereza "
              f"clips, folded {n_conf} confuser clips into Background")
    if args.drop_pogonias:
        n_pog = int((index["label"] == "C_pogonias").sum())
        index.loc[index["label"] == "C_pogonias", "label"] = "Background"
        index = index.reset_index(drop=True)
        print(f"Ablation --drop-pogonias: folded {n_pog} pogonias clips "
              f"into Background")
    if args.drop_extra_confuser:
        extra = index["source"] == "colobus_field_fp"
        n_extra = int(extra.sum())
        index = index[~extra].reset_index(drop=True)
        print(f"Ablation --drop-extra-confuser: removed {n_extra} reviewed "
              f"deployment detections from Colobus_confuser")
    index["hour"] = detection_hours(index)
    print(f"Manifest: {len(index)} clips")
    pack_row = index["row"].to_numpy()

    class_names = sorted(index["label"].unique())
    label_id = {c: i for i, c in enumerate(class_names)}
    labels = index["label"].map(label_id).to_numpy()
    groups = np.array([f"{r.label}/{train_module.source_group(r.path)}"
                       for r in index.itertuples()])
    print(f"Classes: {class_names}")
    print(f"Source groups: {len(set(groups))}\n")

    shared_heads = os.path.abspath(
        os.path.join(REPO, "data", "outputs", "v13_heads"))
    if os.path.normcase(head_dir) == os.path.normcase(shared_heads):
        print("  ! --head-dir explicitly targets the legacy shared v13_heads "
              "directory; a fresh run-specific directory is safer")

    run_metadata = {
        "schema": "v13_loso_run/v1",
        "status": "preparing_feature_cache",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": list(sys.argv),
        "script": _file_metadata(__file__, include_sha256=True),
        "arguments": vars(args),
        "input_artifacts": input_metadata,
        "dataset_after_filters_and_ablations": _dataset_summary(index),
        "outputs": {
            "results_csv": out_path,
            "head_dir": head_dir,
            "run_metadata": metadata_path,
        },
        "completed_folds": [],
        "skipped_folds": [],
    }
    _write_run_metadata(metadata_path, run_metadata)

    if args.prepare_cache_only:
        print("Feature cache preparation")
        try:
            feats = feature_cache(args.images, args.cache,
                                  replace_existing=args.replace_cache)
            mmap = getattr(feats, "_mmap", None)
            if mmap is not None:
                mmap.close()
            del feats
            run_metadata["input_artifacts"]["feature_cache"] = _npy_metadata(
                args.cache, "feature cache", expected_rows=packed_rows,
                expected_trailing_shape=FEATURE_CACHE_SHAPE,
                expected_dtype=np.float16)
        except ArtifactIntegrityError as error:
            run_metadata.update(
                status="failed",
                failure=f"feature cache preparation: {error}",
                finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
            _write_run_metadata(metadata_path, run_metadata)
            sys.exit(f"Refusing to prepare cache: {error}")
        run_metadata.update(
            status="feature_cache_ready_unlocked",
            next_step=("Run check_v13_artifacts.py --full-hash --write-lock "
                       "<new-lock> before a verified-only training run."),
            finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        _write_run_metadata(metadata_path, run_metadata)
        print("Feature cache is ready but not yet a reportable training input. "
              "Issue an immutable full-hash artifact lock next.")
        return

    # Training is read-only with respect to the cache. Cache creation/recovery
    # is deliberately isolated in --prepare-cache-only above.
    print("Feature cache")
    if args.unfreeze:
        # The cache is the frozen base's output and fine-tuning stops holding
        # that fixed, so it cannot be reused; everything downstream indexes
        # `feats` by the same row numbers either way.
        feats = np.load(args.images, mmap_mode="r")
        print(f"Fine-tuning: reading the image pack {feats.shape} rather than "
              f"the feature cache")
    else:
        feats = np.load(args.cache, mmap_mode="r")

    stations = sorted(s for s in index["station"].unique() if s)
    if args.folds != "all":
        stations = [s.strip() for s in args.folds.split(",") if s.strip()]

    # The deployed time window, so the gated columns describe the pipeline as
    # it is actually configured rather than an ungated one nobody will run.
    gate = None
    if config.TIME_FILTER_START and config.TIME_FILTER_END:
        def _h(s):
            hh, mm = s.split(":")
            return int(hh) + int(mm) / 60.0
        gate = (_h(config.TIME_FILTER_START), _h(config.TIME_FILTER_END))
        print(f"Time gate: {config.TIME_FILTER_START}-{config.TIME_FILTER_END} "
              f"-- 'gated_*' columns are the deployed configuration\n")

    run_metadata.update(status="running", requested_folds=stations,
                        time_gate=gate,
                        feature_cache_ready_utc=time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    _write_run_metadata(metadata_path, run_metadata)
    os.makedirs(head_dir, exist_ok=True)

    results = []
    for station in stations:
        tr_mask, ev_mask, cal_mask = fold_masks(index, station, args.keep_all_background)
        n_eval = int(ev_mask.sum())
        if not n_eval:
            print(f"{station}: no reviewed detections -- skipped")
            run_metadata["skipped_folds"].append(station)
            _write_run_metadata(metadata_path, run_metadata)
            continue
        print(f"\n=== hold out {station} ===")
        print(f"  train {int(tr_mask.sum())} clips, "
              f"withheld {int((~tr_mask).sum())}, score on {n_eval} reviewed")

        # Positions in `index` -> rows in the image pack and feature cache.
        tr_rows = pack_row[tr_mask]
        groups_tr = groups[tr_mask]
        labels_tr = labels[tr_mask]
        if args.max_train and len(tr_rows) > args.max_train:
            # Sample by group, not by row, so a capped run still cannot put two
            # windows of one recording on opposite sides of the inner split.
            rng = np.random.default_rng(args.seed)
            uniq = sorted(set(groups_tr))
            keep = set(rng.choice(uniq, size=min(len(uniq), args.max_train // 4),
                                  replace=False))
            sel = np.array([g in keep for g in groups_tr])
            tr_rows, groups_tr, labels_tr = tr_rows[sel], groups_tr[sel], labels_tr[sel]
            print(f"  --max-train: capped to {len(tr_rows)} clips "
                  f"({len(keep)} groups) -- SMOKE TEST, not the experiment")
        t0 = time.time()
        head, val_acc, inner_val_rows = train_head(
            feats, tr_rows, labels_tr, groups_tr, class_names, args.epochs,
            args.seed, args.verbose, pooling=args.pooling,
            patience=args.patience, unfreeze=args.unfreeze,
            finetune_epochs=args.finetune_epochs,
            finetune_lr=args.finetune_lr)
        as_truth = (lambda m: index.loc[m, "label"].map(
            lambda l: "call" if l == "Cernic" else "fp"))

        # Calibrate on the reviewed windows that landed in the head's own inner
        # validation split: still not the held-out station, so the threshold is
        # not an oracle, but also not rows the head was fitted on.
        held = np.isin(pack_row, inner_val_rows)
        cal_mask = cal_mask & held
        n_cal = int(cal_mask.sum())
        print(f"   calibration set: {n_cal} reviewed windows from the inner "
              f"validation split, none trained on")
        if n_cal < 30:
            print("   ! too few to fit a threshold on; the loso_* columns for "
                  "this fold will be absent rather than wrong")

        row = score_fold(head, feats, pack_row[ev_mask], as_truth(ev_mask),
                         class_names, config.DETECTION_CONFIDENCE_THRESHOLD,
                         hours=index.loc[ev_mask, "hour"].to_numpy(),
                         gate=gate,
                         cal_rows=pack_row[cal_mask],
                         cal_truth=as_truth(cal_mask),
                         cal_hours=index.loc[cal_mask, "hour"].to_numpy())
        row.update(station=station, grouped_val_accuracy=round(val_acc, 4),
                   minutes=round((time.time() - t0) / 60, 1))
        results.append(row)

        # Write after every fold. The sweep is 16 folds over a 25 GB memmap and
        # an OOM or a stray Ctrl-C at fold 15 used to discard the other 14.
        _write_csv_atomic(pd.DataFrame(results), out_path)
        _save_weights_atomic(
            head, os.path.join(head_dir, f"head_{station}.weights.h5"))
        run_metadata["completed_folds"].append(station)
        run_metadata["last_completed_fold_utc"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _write_run_metadata(metadata_path, run_metadata)
        print(f"  grouped val acc {val_acc:.4f}")
        print(f"  at the deployment threshold: precision "
              f"{row['v12_precision']:.3f} -> {row['v13_precision']}, "
              f"calls kept {row['calls_retained']}, "
              f"FPs removed {row['fps_removed']}")
        if "matched_precision" in row:
            print(f"  at {100 * 0.95:.0f}% recall (score >= "
                  f"{row['matched_threshold']}): precision "
                  f"{row['matched_precision']}, "
                  f"FPs removed {row['matched_fps_removed']}")

    if not results:
        run_metadata.update(status="failed", failure="no fold produced a score",
                            finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                     time.gmtime()))
        _write_run_metadata(metadata_path, run_metadata)
        sys.exit("no fold produced a score")

    cols = ["station", "detections", "calls", "false_positives", "v12_precision",
            "kept_calls", "kept_false_positives", "calls_retained", "fps_removed",
            "v13_precision", "review_reduction", "matched_threshold",
            "matched_calls_retained", "matched_fps_removed", "matched_precision",
            "gated_detections", "gated_v12_precision", "gated_matched_threshold",
            "gated_matched_calls_retained", "gated_matched_fps_removed",
            "gated_matched_precision",
            # The deployable operating point. gated_loso_* is the pair to quote:
            # threshold fitted on the other fifteen stations, deployed argmax
            # left on, inside the time window the pipeline actually uses.
            "calibration_windows",
            "loso_threshold", "loso_calls_retained", "loso_fps_removed",
            "loso_precision",
            "gated_loso_threshold", "gated_loso_calls_retained",
            "gated_loso_fps_removed", "gated_loso_precision",
            "grouped_val_accuracy", "minutes"]
    df = pd.DataFrame(results)
    df = df[[c for c in cols if c in df.columns]]
    _write_csv_atomic(df, out_path)
    run_metadata.update(
        status="complete",
        finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        result_csv=_file_metadata(args.out, include_sha256=True),
        result_rows=int(len(df)),
    )
    _write_run_metadata(metadata_path, run_metadata)
    print(f"\n{df.to_string(index=False)}")

    tot_call = df["calls"].sum()
    tot_fp = df["false_positives"].sum()
    kept_call = df["kept_calls"].sum()
    kept_fp = df["kept_false_positives"].sum()
    print(f"\nPooled over {len(df)} held-out stations:")
    print(f"  V12 precision {tot_call / (tot_call + tot_fp):.3f}  "
          f"({tot_call} calls / {tot_call + tot_fp} detections)")
    if kept_call + kept_fp:
        print(f"  V13 precision {kept_call / (kept_call + kept_fp):.3f}  "
              f"({kept_call} calls / {kept_call + kept_fp} detections)")
    print(f"  calls retained {kept_call / tot_call:.3f}, "
          f"false positives removed {1 - kept_fp / tot_fp:.3f}")
    # Pooled sums are dominated by whichever station reviewed the most windows:
    # IPA4ST alone is 2 470 of the 6 189 rows and 2 370 of the false positives,
    # almost all of them one nocturnal insect chorus. An unweighted mean over
    # stations answers "how does this do at a typical station", which is the
    # question the paper is actually asking.
    if len(df) > 1:
        print("\nMacro-average over stations (each station counts once):")
        for c in ["v12_precision", "matched_precision", "matched_fps_removed",
                  "matched_calls_retained", "gated_v12_precision",
                  "gated_matched_precision", "gated_matched_fps_removed",
                  "loso_precision", "loso_calls_retained", "loso_fps_removed",
                  "gated_loso_precision", "gated_loso_calls_retained",
                  "gated_loso_fps_removed"]:
            if c in df and df[c].notna().any():
                print(f"  {c:32s} {df[c].astype(float).mean():.4f}"
                      f"   (n={int(df[c].notna().sum())} stations)")

    if "gated_matched_precision" in df and df["gated_matched_precision"].notna().any():
        print("\nRead the gated_* columns. The ungated ones score V13 on the "
              "2 618\nnocturnal windows the deployed time filter already "
              "discards for free,\nwhich credits the model for rejecting an "
              "insect chorus that costs nothing.")

    print(f"\nWrote {args.out}")
    print(f"Per-fold head weights in {head_dir}")
    print(f"Run metadata in {metadata_path}")
    print("\nThis is precision only. Calls V12 never fired on are not in this "
          "set\nand cannot be scored here -- recall needs a re-detection pass.")


if __name__ == "__main__":
    main()
