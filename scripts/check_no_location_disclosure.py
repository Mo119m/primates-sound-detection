"""Fail when a tracked file explains WHERE the recorder positions can be found.

The project has one hard rule: no station coordinate reaches anything public,
because C. guereza is hunted and a recorder position is a place to wait for one.
The guard built for that rule searches for the coordinate pattern itself, and on
2026-08-30 a submission audit found what that guard is structurally unable to
see -- two sentences in the manuscript that between them named the file field
the positions are written into, said how many stations carry it, and listed the
stations that do not, which identifies the rest by elimination. Neither sentence
contained a coordinate, so neither tripped the regex, and they sat beside a Data
availability statement offering the raw recordings on request. A pointer to the
data is the same disclosure as the data.

Those sentences are NOT quoted here. An earlier version of this file reproduced
them in full to explain itself, which put the payload back into the public
repository inside the artefact documenting its removal, and then exempted itself
from its own check. The shape is enough: a position word beside a place-to-find-
it word, or beside a station identifier.

So this looks for the pointer rather than the payload. It is a heuristic and it
errs loud, which is the right direction for a rule whose other failure mode
cannot be undone. Every hit is either rewritten or added to ALLOWED with a
reason.

    python scripts/check_no_location_disclosure.py [path ...]
"""
import os
import re
import subprocess
import sys

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

POSITION = r"(lat/lon|latitude|longitude|\bGPS\b|coordinates?|geotag|deployment position|recorder position)"
LOCATOR = r"(filename|file name|in the name|metadata|EXIF|header|stamp|written into|encoded in|embedded)"
STATION = r"\bIPA\d+(ST)?\b"

# ipynb is deliberately NOT skipped. It was, until the same audit found a
# third instance of the pointer chain in a notebook markdown cell that no
# one had decided to exempt -- an extension list had excluded every
# notebook in the repository silently.
SKIP_EXT = {"pdf", "png", "jpg", "jpeg", "npy", "h5", "zip", "docx", "wav",
            "pyc", "ico", "gz"}

# Exemptions, each a decision on the record rather than a convenience.
#
# The distinction that matters is not "code versus prose", it is what the
# reader of that file can do with the sentence. The pipeline has to strip a
# coordinate out of every filename, and it cannot strip what it cannot name --
# so the implementation necessarily says where the coordinates are, and a
# reader of the code without the audio is no closer to a recorder for knowing
# it. The manuscript is the opposite case: it is the artefact that is indexed
# and read, and it sat beside a Data availability statement offering the raw
# recordings on request, which is what made the same sentence an instruction.
ALLOWED = [
    # This file, and narrowly: its POSITION/LOCATOR patterns and the reasons
    # below necessarily contain the trigger words, so it matches itself four
    # times over its own machinery. A blanket exemption is what let an earlier
    # docstring carry the disclosure verbatim, so the exemption is paired with
    # _assert_no_payload_here() -- the specific thing it could hide is checked
    # explicitly, every run, before anything else.
    "scripts/check_no_location_disclosure.py",
    # Implements the stripping. station_of() needs the no-GPS station list to
    # attribute folds correctly, and _canon() needs the coordinate pattern to
    # remove it; both are the mechanism by which coordinates stay out of every
    # derived artefact.
    "scripts/build_v13_dataset.py",
    # Builds the shared review table by canonicalising the same filenames. Its
    # whole purpose is that the shared file carries three columns and no
    # coordinate.
    "scripts/make_review_gate_table.py",
    # Describes the fold-attribution consequence for a Colab runner.
    "colab/make_notebook.py",
    # Same mechanism, training side: possible_stations() has to know which
    # filenames narrow to a group, and _canon_review_key() has to strip the
    # coordinate to join the review table.
    "scripts/train_v13_loso.py",
    # The spatial-extent control itself. It reads positions in order to compute
    # inter-recorder distances, and it is the reason the manuscript can say a
    # roar carrying 1.6 km cannot reach two recorders 4.7 km apart. It reads
    # them from audio that is not in this repository and prints no absolute
    # position.
    "scripts/spatial_extent_test.py",
    # A session note recording that per-station subfolders attribute 253 clips
    # where the filename attributes 156. It names no station and points at a
    # directory layout on a drive that is not published.
    "docs/history/SESSION_2026-08-03.md",
]


def _assert_no_payload_here():
    """This file must describe the shape and never restate the sentences.

    Paired with this file's own ALLOWED entry. The exemption is needed -- the
    patterns above match themselves -- but on 2026-08-31 an audit found that an
    earlier docstring had used that exemption to carry the two retired
    sentences in full, in a public repository, inside the artefact documenting
    their removal. So the exemption buys silence on the machinery only, and
    these fragments are refused by name.
    """
    # The list below is the definition of what is forbidden, so it necessarily
    # contains it. Searching the whole file would therefore always fire -- the
    # first version of this function did exactly that. Everything between the
    # two markers is excluded from the search, and nothing else is.
    fragments = [                                    # BEGIN-FORBIDDEN-LIST
        "stamp a unique lat",
        "recovers coordinates for 11",
        "writes the deployment lat",
        "recorded with GPS disabled",
        "recorded with GPS off",
    ]                                                # END-FORBIDDEN-LIST
    raw = open(os.path.abspath(__file__), encoding="utf-8").read()
    a, b = raw.find("BEGIN-FORBIDDEN-LIST"), raw.find("END-FORBIDDEN-LIST")
    if a < 0 or b < 0:
        sys.exit("the forbidden-list markers are gone; this check is disarmed")
    text = (raw[:a] + raw[b:]).lower()
    found = [f for f in fragments if f.lower() in text]
    if found:
        sys.exit(
            "this file restates the disclosure it exists to catch: "
            + ", ".join(repr(f) for f in found)
            + "\n  Describe the sentence shape instead. The exemption in "
              "ALLOWED silences the machinery, not the payload.")


def sentences(text):
    for para in text.split("\n\n"):
        for s in re.split(r"(?<=[.!?])\s+", para.replace("\n", " ")):
            if s.strip():
                yield s.strip()


def main():
    _assert_no_payload_here()
    targets = sys.argv[1:]
    if not targets:
        targets = subprocess.run(["git", "-C", REPO, "ls-files"],
                                 capture_output=True, text=True).stdout.split()
    hits = []
    scanned = 0
    for f in targets:
        if f.rsplit(".", 1)[-1].lower() in SKIP_EXT or f in ALLOWED:
            continue
        p = f if os.path.isabs(f) else os.path.join(REPO, f)
        try:
            text = open(p, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        scanned += 1
        for s in sentences(text):
            pos = re.search(POSITION, s, re.I)
            if not pos:
                continue
            loc = re.search(LOCATOR, s, re.I)
            sta = re.search(STATION, s)
            if loc or sta:
                why = "position + locator" if loc else "position + station id"
                hits.append((f, why, s[:190]))

    print(f"  files scanned: {scanned}")
    if not hits:
        print("  no sentence points at where recorder positions can be found")
        return 0
    print(f"  {len(hits)} sentence(s) to review:\n")
    for f, why, s in hits:
        print(f"  {f}  [{why}]")
        print(f"    {s}\n")
    print("  Each is either rewritten so it does not locate the array, or added\n"
          "  to ALLOWED in this file with a reason. A pointer to the data is the\n"
          "  same disclosure as the data.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
