#!/usr/bin/env python3
"""Fail closed when V13 training artifacts do not describe one dataset.

The V13 training path has four coupled artifacts:

* ``v13_manifest.csv`` is the current, auditable list of labelled clips.
* ``v13_index.csv`` is the manifest that was actually packed, plus the image
  row address and render status.
* ``v13_images.npy`` holds one 224 x 224 x 3 uint8 image for every index row.
* ``v13_features.npy`` holds the VGG19 ``block4_conv4`` activation for every
  image row, as float16 (28 x 28 x 512).

The training code can intentionally filter an older index by a newer manifest,
but that is not sufficient evidence that the image and feature artifacts still
refer to the current labels.  This checker is deliberately stricter: unless
every manifest/index row agrees in order and all array contracts agree with the
index, it exits non-zero.  Run it before a costly LOSO sweep.

Examples
--------
    python scripts/check_v13_artifacts.py
    python scripts/check_v13_artifacts.py --json
    python scripts/check_v13_artifacts.py --full-hash --write-lock data/outputs/v13.lock.json
    python scripts/check_v13_artifacts.py --full-hash --lock data/outputs/v13.lock.json

The default mode is read-only.  ``--json-out`` and ``--write-lock`` are the
only options that write files, and both require an explicit path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import posixpath
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO = Path(__file__).resolve().parent.parent
CHECKER_VERSION = "1.0"
LOCK_SCHEMA_VERSION = 1
LOCK_KIND = "v13-artifact-lock"

MANIFEST_FIELDS = (
    "path",
    "label",
    "station",
    "possible_stations",
    "source",
    "verified",
    "period",
    "aug",
)
INDEX_FIELDS = MANIFEST_FIELDS + ("row", "ok")
PATH_INVARIANT_FIELDS = (
    "label",
    "station",
    "possible_stations",
    "source",
    "verified",
    "period",
)
DEFAULT_IMAGE_SHAPE = (224, 224, 3)
DEFAULT_CACHE_SHAPE = (28, 28, 512)
TRUE_VALUES = {"1", "true", "t", "yes", "y"}
FALSE_VALUES = {"0", "false", "f", "no", "n"}


class Issues:
    """Collect bounded, JSON-serialisable validation errors and warnings."""

    def __init__(self, max_examples: int = 20) -> None:
        self.errors: list[dict[str, Any]] = []
        self.warnings: list[dict[str, Any]] = []
        self.max_examples = max_examples

    def error(self, code: str, message: str, **details: Any) -> None:
        if len(self.errors) < self.max_examples:
            issue: dict[str, Any] = {"code": code, "message": message}
            if details:
                issue["details"] = details
            self.errors.append(issue)

    def warning(self, code: str, message: str, **details: Any) -> None:
        if len(self.warnings) < self.max_examples:
            issue: dict[str, Any] = {"code": code, "message": message}
            if details:
                issue["details"] = details
            self.warnings.append(issue)


def _path_arg(value: str) -> Path:
    return Path(value).expanduser()


def parse_shape(value: str) -> tuple[int, ...]:
    """Parse a trailing array shape such as ``224,224,3`` or ``28x28x512``."""
    text = value.lower().replace("x", ",")
    try:
        shape = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid shape {value!r}; use comma- or x-separated positive integers"
        ) from exc
    if not shape or any(dimension <= 0 for dimension in shape):
        raise argparse.ArgumentTypeError(
            f"invalid shape {value!r}; dimensions must be positive integers"
        )
    return shape


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    outputs = REPO / "data" / "outputs"
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifest", type=_path_arg,
                        default=outputs / "v13_manifest.csv")
    parser.add_argument("--index", type=_path_arg,
                        default=outputs / "v13_index.csv")
    parser.add_argument("--images", type=_path_arg,
                        default=outputs / "v13_images.npy")
    parser.add_argument("--cache", type=_path_arg,
                        default=outputs / "v13_features.npy")
    parser.add_argument(
        "--image-shape", type=parse_shape, default=DEFAULT_IMAGE_SHAPE,
        metavar="H,W,C",
        help="expected image dimensions after the row dimension (default: 224,224,3)",
    )
    parser.add_argument(
        "--cache-shape", type=parse_shape, default=DEFAULT_CACHE_SHAPE,
        metavar="H,W,C",
        help="expected VGG feature dimensions after the row dimension "
             "(default: 28,28,512)",
    )
    parser.add_argument(
        "--sample-rows", type=int, default=3, metavar="N",
        help="number of renderable rows sampled for finite feature values "
             "(default: 3; use 0 to skip)",
    )
    parser.add_argument(
        "--full-hash", action="store_true",
        help="also calculate full SHA-256 values for the large NPY files; slow",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="write the report as JSON to stdout instead of human-readable text",
    )
    parser.add_argument(
        "--json-out", type=_path_arg, metavar="PATH",
        help="also write the JSON report to a new PATH (explicit write; refuses "
             "to overwrite unless --overwrite-json-out is supplied)",
    )
    parser.add_argument(
        "--overwrite-json-out", action="store_true",
        help="allow --json-out to replace an existing report",
    )
    lock_group = parser.add_mutually_exclusive_group()
    lock_group.add_argument(
        "--lock", type=_path_arg, metavar="PATH",
        help="validate current artifacts against a previously written immutable lock",
    )
    lock_group.add_argument(
        "--write-lock", type=_path_arg, metavar="PATH",
        help="write a new lock only if all checks pass; refuses to overwrite",
    )
    args = parser.parse_args(argv)
    if args.sample_rows < 0:
        parser.error("--sample-rows must be zero or greater")
    if args.write_lock and not args.full_hash:
        parser.error("--write-lock requires --full-hash so a lock is immutable")
    return args


def normalise_path(value: str) -> str:
    """Make path comparison stable across CSV slash styles on the current OS."""
    text = value.strip().replace("\\", "/")
    if not text:
        return ""
    # Keep a relative path relative.  ``posixpath.normpath`` is intentional:
    # CSV paths are stored with forward slashes even when the pack was made on
    # Windows, and resolving would fail for clips on an external drive.
    normalised = posixpath.normpath(text)
    if os.name == "nt":
        normalised = normalised.casefold()
    return normalised


def normalise_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def parse_bool(value: Any) -> bool | None:
    text = normalise_text(value).casefold()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    return None


def parse_nonnegative_int(value: Any) -> int | None:
    text = normalise_text(value)
    if not text:
        return None
    try:
        parsed = int(text)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def split_stations(value: Any) -> tuple[str, ...] | None:
    text = normalise_text(value)
    if not text:
        return ()
    parts = tuple(part.strip() for part in text.split(";"))
    if not all(parts) or len(set(parts)) != len(parts):
        return None
    return parts


def canonical_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, tuple):
        return ";".join(value)
    return str(value)


def read_csv_artifact(path: Path, name: str, issues: Issues) -> tuple[list[str], list[dict[str, str]]]:
    """Read a small CSV without coercing data types that are part of identity."""
    if not path.is_file():
        issues.error("missing_file", f"{name} file is missing", path=str(path))
        return [], []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            if not fieldnames:
                issues.error("empty_csv", f"{name} has no header", path=str(path))
                return [], []
            fields = [normalise_text(field) for field in fieldnames]
            if any(not field for field in fields):
                issues.error("blank_csv_header", f"{name} has a blank CSV header", path=str(path))
            duplicate_fields = sorted(field for field, count in Counter(fields).items()
                                      if field and count > 1)
            if duplicate_fields:
                issues.error("duplicate_csv_header", f"{name} has duplicate CSV headers",
                             path=str(path), fields=duplicate_fields)
            rows: list[dict[str, str]] = []
            malformed_rows = 0
            for line_number, raw in enumerate(reader, start=2):
                if None in raw:
                    malformed_rows += 1
                    if malformed_rows <= 3:
                        issues.error(
                            "csv_extra_columns",
                            f"{name} row has more values than its header",
                            path=str(path), line=line_number,
                        )
                rows.append({field: normalise_text(raw.get(field)) for field in fields})
            if malformed_rows > 3:
                issues.error("csv_extra_columns_count",
                             f"{name} has rows with more values than its header",
                             path=str(path), count=malformed_rows)
            return fields, rows
    except (OSError, UnicodeError, csv.Error) as exc:
        issues.error("unreadable_csv", f"cannot read {name}", path=str(path), error=str(exc))
        return [], []


def require_fields(fields: Iterable[str], expected: Iterable[str], name: str,
                   path: Path, issues: Issues) -> bool:
    missing = [field for field in expected if field not in fields]
    if missing:
        issues.error("missing_columns", f"{name} is missing required columns",
                     path=str(path), columns=missing)
        return False
    return True


def canonicalise_rows(rows: list[dict[str, str]], fields: Sequence[str], name: str,
                      issues: Issues) -> list[dict[str, Any]]:
    """Validate V13 row metadata and return canonical identity/provenance rows."""
    canonical: list[dict[str, Any]] = []
    invalid_bool: list[int] = []
    invalid_aug: list[int] = []
    invalid_row: list[int] = []
    invalid_ok: list[int] = []
    bad_station_list: list[int] = []
    station_mismatch: list[int] = []
    blank_path: list[int] = []
    blank_label: list[int] = []
    blank_source: list[int] = []

    has_row = "row" in fields
    has_ok = "ok" in fields
    for position, raw in enumerate(rows):
        path = normalise_path(raw.get("path", ""))
        label = normalise_text(raw.get("label", ""))
        station = normalise_text(raw.get("station", ""))
        possible_stations = split_stations(raw.get("possible_stations", ""))
        source = normalise_text(raw.get("source", ""))
        period = normalise_text(raw.get("period", ""))
        verified = parse_bool(raw.get("verified", ""))
        aug = parse_nonnegative_int(raw.get("aug", ""))
        row = parse_nonnegative_int(raw.get("row", "")) if has_row else None
        ok = parse_bool(raw.get("ok", "")) if has_ok else None

        line = position + 2
        if not path:
            blank_path.append(line)
        if not label:
            blank_label.append(line)
        if not source:
            blank_source.append(line)
        if possible_stations is None:
            bad_station_list.append(line)
            possible_stations = ()
        if station and station not in possible_stations:
            station_mismatch.append(line)
        if verified is None:
            invalid_bool.append(line)
            verified = False
        if aug is None:
            invalid_aug.append(line)
            aug = -1
        if has_row and row is None:
            invalid_row.append(line)
            row = -1
        if has_ok and ok is None:
            invalid_ok.append(line)
            ok = False

        record: dict[str, Any] = {
            "path": path,
            "label": label,
            "station": station,
            "possible_stations": possible_stations,
            "source": source,
            "verified": verified,
            "period": period,
            "aug": aug,
        }
        if has_row:
            record["row"] = row
        if has_ok:
            record["ok"] = ok
        canonical.append(record)

    def report_lines(code: str, message: str, lines: list[int]) -> None:
        if lines:
            issues.error(code, message, count=len(lines), lines=lines[:5])

    report_lines("blank_path", f"{name} has blank clip paths", blank_path)
    report_lines("blank_label", f"{name} has blank labels", blank_label)
    report_lines("blank_source", f"{name} has blank provenance sources", blank_source)
    report_lines("invalid_possible_stations",
                 f"{name} has empty or duplicate possible_stations values", bad_station_list)
    report_lines("station_not_in_possible_stations",
                 f"{name} station is not included in possible_stations", station_mismatch)
    report_lines("invalid_verified", f"{name} has non-boolean verified values", invalid_bool)
    report_lines("invalid_aug", f"{name} has invalid augmentation indices", invalid_aug)
    report_lines("invalid_row", f"{name} has invalid image row values", invalid_row)
    report_lines("invalid_ok", f"{name} has non-boolean render status values", invalid_ok)
    return canonical


def sample_values(values: Iterable[Any], limit: int = 5) -> list[Any]:
    result: list[Any] = []
    for value in values:
        result.append(value)
        if len(result) >= limit:
            break
    return result


def sample_key(record: dict[str, Any]) -> tuple[str, int]:
    return record["path"], record["aug"]


def signature(record: dict[str, Any], fields: Sequence[str] = MANIFEST_FIELDS) -> tuple[Any, ...]:
    return tuple(record[field] for field in fields)


def validate_duplicates_and_conflicts(records: list[dict[str, Any]], name: str,
                                      issues: Issues) -> None:
    identities = Counter(sample_key(record) for record in records)
    duplicates = [key for key, count in identities.items() if count > 1]
    if duplicates:
        issues.error("duplicate_sample_identity",
                     f"{name} repeats path + aug identities",
                     count=len(duplicates), examples=[list(value) for value in duplicates[:5]])

    by_path: dict[str, tuple[Any, ...]] = {}
    conflicts: list[str] = []
    for record in records:
        path = record["path"]
        provenance = signature(record, PATH_INVARIANT_FIELDS)
        previous = by_path.setdefault(path, provenance)
        if previous != provenance and path not in conflicts:
            conflicts.append(path)
    if conflicts:
        issues.error("conflicting_path_metadata",
                     f"{name} assigns incompatible labels/provenance to one path",
                     count=len(conflicts), examples=conflicts[:5])


def validate_index_rows(records: list[dict[str, Any]], issues: Issues) -> None:
    if not records:
        return
    rows = [record.get("row", -1) for record in records]
    duplicate_rows = [row for row, count in Counter(rows).items() if count > 1]
    if duplicate_rows:
        issues.error("duplicate_image_row", "index reuses image row addresses",
                     count=len(duplicate_rows), examples=duplicate_rows[:5])
    expected = list(range(len(records)))
    if rows != expected:
        mismatches = [
            {"index_position": position, "row": row, "expected": position}
            for position, row in enumerate(rows) if row != position
        ]
        issues.error("noncontiguous_or_reordered_rows",
                     "index row values must be exactly 0..N-1 in index order",
                     count=len(mismatches), examples=mismatches[:5])


def compare_manifest_and_index(manifest: list[dict[str, Any]],
                               index: list[dict[str, Any]], issues: Issues) -> None:
    if len(manifest) != len(index):
        issues.error("row_count_mismatch", "manifest and index row counts differ",
                     manifest_rows=len(manifest), index_rows=len(index))

    manifest_ids = Counter(sample_key(record) for record in manifest)
    index_ids = Counter(sample_key(record) for record in index)
    if manifest_ids != index_ids:
        missing = list((manifest_ids - index_ids).elements())
        extra = list((index_ids - manifest_ids).elements())
        issues.error("sample_identity_set_mismatch",
                     "manifest and index do not contain the same path + aug samples",
                     missing_from_index=[list(value) for value in missing[:5]],
                     extra_in_index=[list(value) for value in extra[:5]],
                     missing_count=sum((manifest_ids - index_ids).values()),
                     extra_count=sum((index_ids - manifest_ids).values()))

    manifest_paths = {record["path"] for record in manifest}
    index_paths = {record["path"] for record in index}
    if manifest_paths != index_paths:
        issues.error("path_set_mismatch", "manifest and index do not contain the same paths",
                     missing_from_index=sample_values(sorted(manifest_paths - index_paths)),
                     extra_in_index=sample_values(sorted(index_paths - manifest_paths)),
                     missing_count=len(manifest_paths - index_paths),
                     extra_count=len(index_paths - manifest_paths))

    mismatches: list[dict[str, Any]] = []
    for position, (manifest_record, index_record) in enumerate(zip(manifest, index)):
        if signature(manifest_record) != signature(index_record):
            changed = [
                field for field in MANIFEST_FIELDS
                if manifest_record[field] != index_record[field]
            ]
            if len(mismatches) < 5:
                mismatches.append({
                    "position": position,
                    "fields": changed,
                    "manifest_path": manifest_record["path"],
                    "index_path": index_record["path"],
                })
    if mismatches:
        # Count again without retaining every mismatching row in memory.
        mismatch_count = sum(
            signature(manifest_record) != signature(index_record)
            for manifest_record, index_record in zip(manifest, index)
        )
        issues.error("manifest_index_order_or_metadata_mismatch",
                     "index must preserve every manifest row and its metadata in order",
                     count=mismatch_count, examples=mismatches)


def canonical_rows_sha256(records: list[dict[str, Any]], fields: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(b"v13-canonical-rows-v1\n")
    for record in records:
        encoded = "\x1f".join(canonical_scalar(record[field]) for field in fields)
        digest.update(encoded.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sample_file_sha256(path: Path, chunk_size: int = 64 * 1024) -> str:
    """A stable, documented fingerprint for multi-gigabyte NPY artifacts."""
    size = path.stat().st_size
    offsets = sorted({
        0,
        max(0, (size - chunk_size) // 2),
        max(0, size - chunk_size),
    })
    digest = hashlib.sha256()
    digest.update(b"v13-file-sample-v1\n")
    with path.open("rb") as handle:
        for offset in offsets:
            handle.seek(offset)
            block = handle.read(min(chunk_size, size - offset))
            digest.update(f"{offset}:{len(block)}\n".encode("ascii"))
            digest.update(block)
    return digest.hexdigest()


def close_memmap(array: Any) -> None:
    mmap = getattr(array, "_mmap", None)
    if mmap is not None:
        mmap.close()


def evenly_spaced(values: Sequence[int], count: int) -> list[int]:
    if count <= 0 or not values:
        return []
    if len(values) <= count:
        return list(values)
    if count == 1:
        return [values[len(values) // 2]]
    return [values[round(i * (len(values) - 1) / (count - 1))] for i in range(count)]


def inspect_npy(path: Path, name: str, expected_rows: int | None,
                expected_shape: tuple[int, ...], expected_dtype: str,
                sample_rows: Sequence[int], full_hash: bool, numpy: Any,
                issues: Issues) -> dict[str, Any]:
    """Check an NPY header, its physical byte length, and selected cache rows."""
    summary: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    partial = Path(str(path) + ".partial")
    if partial.exists():
        issues.error("interrupted_artifact_present",
                     f"interrupted {name} artifact is present beside the live file",
                     path=str(partial), bytes=partial.stat().st_size)
        summary["partial_path"] = str(partial)
        summary["partial_bytes"] = partial.stat().st_size
    if not path.is_file():
        issues.error("missing_file", f"{name} file is missing", path=str(path))
        return summary

    try:
        array = numpy.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:  # numpy uses several exception types for malformed NPYs.
        issues.error("unreadable_npy", f"cannot open {name} as a safe NPY array",
                     path=str(path), error=str(exc))
        return summary

    try:
        shape = tuple(int(dimension) for dimension in array.shape)
        dtype = str(array.dtype)
        file_bytes = path.stat().st_size
        offset = getattr(array, "offset", None)
        summary.update({
            "shape": list(shape),
            "dtype": dtype,
            "file_bytes": file_bytes,
            "npy_offset": int(offset) if offset is not None else None,
            "sample_sha256": sample_file_sha256(path),
        })
        if full_hash:
            summary["full_sha256"] = hash_file(path)

        if len(shape) != len(expected_shape) + 1:
            issues.error("npy_rank_mismatch", f"{name} has the wrong array rank",
                         path=str(path), actual_shape=list(shape),
                         expected_rank=len(expected_shape) + 1)
        else:
            if expected_rows is not None and shape[0] != expected_rows:
                issues.error("npy_row_count_mismatch",
                             f"{name} row count does not match the index",
                             path=str(path), actual_rows=shape[0],
                             expected_rows=expected_rows)
            if tuple(shape[1:]) != expected_shape:
                issues.error("npy_shape_mismatch", f"{name} has the wrong trailing shape",
                             path=str(path), actual_shape=list(shape[1:]),
                             expected_shape=list(expected_shape))
        expected_numpy_dtype = numpy.dtype(expected_dtype)
        if array.dtype != expected_numpy_dtype:
            issues.error("npy_dtype_mismatch", f"{name} has the wrong dtype",
                         path=str(path), actual_dtype=dtype,
                         expected_dtype=str(expected_numpy_dtype))

        if offset is not None:
            expected_file_bytes = int(offset) + int(array.size) * int(array.dtype.itemsize)
            summary["expected_file_bytes"] = expected_file_bytes
            if file_bytes != expected_file_bytes:
                issues.error("npy_file_size_mismatch",
                             f"{name} file size does not match its NPY header",
                             path=str(path), actual_bytes=file_bytes,
                             expected_bytes=expected_file_bytes)

        valid_samples = [row for row in sample_rows if 0 <= row < len(array)]
        sampled: list[int] = []
        nonfinite: list[int] = []
        # Integer image arrays are finite by construction.  Testing floating
        # feature rows catches a partially written or numerically corrupt cache
        # without reading tens of GB.
        if numpy.issubdtype(array.dtype, numpy.floating):
            for row in valid_samples:
                sampled.append(row)
                if not numpy.isfinite(array[row]).all():
                    nonfinite.append(row)
            if nonfinite:
                issues.error("nonfinite_feature_sample",
                             f"{name} has non-finite values in sampled renderable rows",
                             path=str(path), rows=nonfinite)
        summary["sampled_rows"] = sampled
    finally:
        close_memmap(array)
    return summary


def count_by(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts = Counter(canonical_scalar(record[key]) for record in records)
    return dict(sorted(counts.items()))


def count_source_groups(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(record["source"].split(":", 1)[0] for record in records)
    return dict(sorted(counts.items()))


def check_lock(lock_path: Path, fingerprints: dict[str, Any], full_hash: bool,
               issues: Issues) -> dict[str, Any]:
    status: dict[str, Any] = {"path": str(lock_path), "mode": "validate"}
    if not lock_path.is_file():
        issues.error("missing_lock", "immutable artifact lock file is missing", path=str(lock_path))
        return status
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        issues.error("unreadable_lock", "cannot read immutable artifact lock",
                     path=str(lock_path), error=str(exc))
        return status
    if (lock.get("kind") != LOCK_KIND or
            lock.get("schema_version") != LOCK_SCHEMA_VERSION):
        issues.error("lock_schema_mismatch", "artifact lock has an unsupported schema",
                     path=str(lock_path), actual={"kind": lock.get("kind"),
                     "schema_version": lock.get("schema_version")},
                     expected={"kind": LOCK_KIND,
                     "schema_version": LOCK_SCHEMA_VERSION})
        return status
    expected = lock.get("fingerprints")
    if not isinstance(expected, dict):
        issues.error("invalid_lock", "artifact lock has no fingerprints object", path=str(lock_path))
        return status

    mismatches: list[dict[str, Any]] = []
    fields_by_artifact = {
        "manifest": ("file_sha256", "canonical_rows_sha256"),
        "index": ("file_sha256", "canonical_rows_sha256"),
        "images": ("shape", "dtype", "file_bytes", "sample_sha256", "full_sha256"),
        "cache": ("shape", "dtype", "file_bytes", "sample_sha256", "full_sha256"),
    }
    for artifact, fields in fields_by_artifact.items():
        expected_artifact = expected.get(artifact)
        current_artifact = fingerprints.get(artifact)
        if not isinstance(expected_artifact, dict) or not isinstance(current_artifact, dict):
            mismatches.append({"artifact": artifact, "field": "artifact", "reason": "missing"})
            continue
        for field in fields:
            if field not in expected_artifact:
                if field == "full_sha256" and full_hash:
                    mismatches.append({
                        "artifact": artifact,
                        "field": field,
                        "reason": "missing from lock; issue a new full-hash lock",
                    })
                continue
            if field == "full_sha256" and not full_hash:
                # Full hashes are cheap to compare but expensive to calculate;
                # a lock that requires one makes it an explicit caller choice.
                mismatches.append({
                    "artifact": artifact,
                    "field": field,
                    "reason": "rerun with --full-hash",
                })
                continue
            if expected_artifact.get(field) != current_artifact.get(field):
                mismatches.append({
                    "artifact": artifact,
                    "field": field,
                    "expected": expected_artifact.get(field),
                    "actual": current_artifact.get(field),
                })
    if mismatches:
        issues.error("artifact_lock_mismatch",
                     "current artifact fingerprints differ from the immutable lock",
                     count=len(mismatches), examples=mismatches[:5])
    else:
        status["matches"] = True
    return status


def write_json_atomic(path: Path, payload: dict[str, Any], *, overwrite: bool = True) -> None:
    """Write an explicitly requested report/lock without leaving a partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial",
                                           dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def lock_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": LOCK_KIND,
        "schema_version": LOCK_SCHEMA_VERSION,
        "checker_version": CHECKER_VERSION,
        "contract": report["contract"],
        "fingerprints": report["fingerprints"],
    }


def run_checks(args: argparse.Namespace) -> dict[str, Any]:
    issues = Issues()
    report: dict[str, Any] = {
        "checker": "check_v13_artifacts.py",
        "checker_version": CHECKER_VERSION,
        "contract": {
            "manifest_fields": list(MANIFEST_FIELDS),
            "index_fields": list(INDEX_FIELDS),
            "image_shape_after_rows": list(args.image_shape),
            "image_dtype": "uint8",
            "cache_shape_after_rows": list(args.cache_shape),
            "cache_dtype": "float16",
        },
        "artifacts": {},
        "fingerprints": {},
    }

    manifest_fields, manifest_raw = read_csv_artifact(args.manifest, "manifest", issues)
    index_fields, index_raw = read_csv_artifact(args.index, "index", issues)
    manifest_valid = require_fields(manifest_fields, MANIFEST_FIELDS, "manifest",
                                    args.manifest, issues)
    index_valid = require_fields(index_fields, INDEX_FIELDS, "index", args.index, issues)
    manifest = canonicalise_rows(manifest_raw, manifest_fields, "manifest", issues) \
        if manifest_valid else []
    index = canonicalise_rows(index_raw, index_fields, "index", issues) if index_valid else []

    if manifest_valid:
        validate_duplicates_and_conflicts(manifest, "manifest", issues)
    if index_valid:
        validate_duplicates_and_conflicts(index, "index", issues)
        validate_index_rows(index, issues)
    if manifest_valid and index_valid:
        compare_manifest_and_index(manifest, index, issues)

    report["artifacts"]["manifest"] = {
        "path": str(args.manifest),
        "rows": len(manifest_raw),
        "unique_paths": len({record["path"] for record in manifest}),
        "augmentation_counts": count_by(manifest, "aug") if manifest_valid else {},
        "label_counts": count_by(manifest, "label") if manifest_valid else {},
        "source_group_counts": count_source_groups(manifest) if manifest_valid else {},
    }
    report["artifacts"]["index"] = {
        "path": str(args.index),
        "rows": len(index_raw),
        "unique_paths": len({record["path"] for record in index}),
        "renderable_rows": sum(record.get("ok") is True for record in index),
        "failed_render_rows": sum(record.get("ok") is False for record in index),
        "augmentation_counts": count_by(index, "aug") if index_valid else {},
        "label_counts": count_by(index, "label") if index_valid else {},
        "source_group_counts": count_source_groups(index) if index_valid else {},
    }

    for artifact, path, records, fields in (
        ("manifest", args.manifest, manifest, MANIFEST_FIELDS),
        ("index", args.index, index, MANIFEST_FIELDS),
    ):
        fingerprint: dict[str, Any] = {"path": str(path)}
        if path.is_file():
            try:
                fingerprint["file_sha256"] = hash_file(path)
            except OSError as exc:
                issues.error("unhashable_file", f"cannot hash {artifact}",
                             path=str(path), error=str(exc))
        if records:
            fingerprint["canonical_rows_sha256"] = canonical_rows_sha256(records, fields)
        report["fingerprints"][artifact] = fingerprint

    try:
        import numpy  # Imported lazily so --help and CSV-only failures remain useful.
    except ImportError as exc:
        issues.error("missing_dependency", "numpy is required to inspect V13 NPY artifacts",
                     error=str(exc))
        report["artifacts"]["images"] = {"path": str(args.images), "exists": args.images.is_file()}
        report["artifacts"]["cache"] = {"path": str(args.cache), "exists": args.cache.is_file()}
        report["fingerprints"]["images"] = {"path": str(args.images)}
        report["fingerprints"]["cache"] = {"path": str(args.cache)}
    else:
        renderable_rows = [
            record["row"] for record in index
            if record.get("ok") is True and isinstance(record.get("row"), int)
        ]
        samples = evenly_spaced(renderable_rows, args.sample_rows)
        expected_rows = len(index) if index_valid else None
        images = inspect_npy(args.images, "images", expected_rows, tuple(args.image_shape),
                             "uint8", (), args.full_hash, numpy, issues)
        cache = inspect_npy(args.cache, "feature cache", expected_rows,
                            tuple(args.cache_shape), "float16", samples,
                            args.full_hash, numpy, issues)
        report["artifacts"]["images"] = images
        report["artifacts"]["cache"] = cache
        report["fingerprints"]["images"] = {
            key: value for key, value in images.items()
            if key in {"path", "shape", "dtype", "file_bytes", "sample_sha256", "full_sha256"}
        }
        report["fingerprints"]["cache"] = {
            key: value for key, value in cache.items()
            if key in {"path", "shape", "dtype", "file_bytes", "sample_sha256", "full_sha256"}
        }

    if args.lock:
        report["lock"] = check_lock(args.lock, report["fingerprints"], args.full_hash, issues)
    if args.write_lock:
        report["lock"] = {"path": str(args.write_lock), "mode": "write"}
        if issues.errors:
            report["lock"]["written"] = False
            report["lock"]["reason"] = "integrity checks failed; lock was not written"
        else:
            try:
                write_json_atomic(args.write_lock, lock_payload(report), overwrite=False)
            except OSError as exc:
                issues.error("lock_write_failed", "cannot write immutable artifact lock",
                             path=str(args.write_lock), error=str(exc))
                report["lock"]["written"] = False
            else:
                report["lock"]["written"] = True

    report["errors"] = issues.errors
    report["warnings"] = issues.warnings
    report["ok"] = not issues.errors
    return report


def short_hash(value: Any) -> str:
    return str(value)[:12] if value else "-"


def render_human(report: dict[str, Any]) -> str:
    lines = [
        "V13 artifact integrity: " + ("PASS" if report["ok"] else "FAIL"),
        "",
        "Artifact summary:",
    ]
    for name in ("manifest", "index", "images", "cache"):
        artifact = report["artifacts"].get(name, {})
        if name in {"manifest", "index"}:
            lines.append(
                f"  {name:8} rows={artifact.get('rows', '?')} "
                f"unique_paths={artifact.get('unique_paths', '?')} "
                f"aug={artifact.get('augmentation_counts', {})} "
                f"path={artifact.get('path', '?')}"
            )
        else:
            lines.append(
                f"  {name:8} shape={artifact.get('shape', '?')} "
                f"dtype={artifact.get('dtype', '?')} "
                f"bytes={artifact.get('file_bytes', '?')} "
                f"path={artifact.get('path', '?')}"
            )

    lines.extend(["", "Fingerprints:"])
    for name in ("manifest", "index", "images", "cache"):
        fingerprint = report["fingerprints"].get(name, {})
        row_hash = fingerprint.get("canonical_rows_sha256")
        file_hash = fingerprint.get("file_sha256") or fingerprint.get("sample_sha256")
        lines.append(
            f"  {name:8} rows={short_hash(row_hash)} file/sample={short_hash(file_hash)}"
        )

    lock = report.get("lock")
    if lock:
        lines.extend(["", f"Lock: {lock}"])
    if report["errors"]:
        lines.extend(["", f"Errors ({len(report['errors'])} shown):"])
        for issue in report["errors"]:
            details = issue.get("details")
            suffix = f" | {json.dumps(details, ensure_ascii=False, sort_keys=True)}" if details else ""
            lines.append(f"  [{issue['code']}] {issue['message']}{suffix}")
    else:
        lines.extend(["", "All manifest/index/array contracts agree."])
    if report["warnings"]:
        lines.extend(["", f"Warnings ({len(report['warnings'])} shown):"])
        for issue in report["warnings"]:
            lines.append(f"  [{issue['code']}] {issue['message']}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_checks(args)
    if args.json_out:
        try:
            write_json_atomic(args.json_out, report,
                              overwrite=args.overwrite_json_out)
        except OSError as exc:
            report["errors"].append({
                "code": "json_report_write_failed",
                "message": "cannot write requested JSON report",
                "details": {"path": str(args.json_out), "error": str(exc)},
            })
            report["ok"] = False
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_human(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
