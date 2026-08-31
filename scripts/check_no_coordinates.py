"""Fail when a tracked file carries a recorder coordinate, a key, or a token.

This is the guard for the project's one hard rule: C. guereza is hunted, a
recorder position is a place to wait for one, and no station coordinate may
reach anything public. It is the companion to check_no_location_disclosure.py,
which looks for prose POINTING at where the coordinates live; this one looks
for the coordinates themselves.

It lived in a scratch directory until 2026-08-31 and was run by hand. That is
how the rule was actually enforced for three weeks: by someone remembering. It
is in the repository now and wired into CI, so a push that reintroduces a
coordinate fails without anyone having to remember.

Two design notes, both learned rather than chosen.

The coordinate pattern is deliberately width-loose. The strict form used
elsewhere -- two digits of latitude, three of longitude -- is correct for the
filenames it was written to strip, and on 2026-08-31 an audit found it passing
+000.0000+000.0000 straight through a check that claimed to refuse any
coordinate. A guard should be wider than the data it guards. The no-fix
sentinel is exempted by exact value below, never by width: exempting a value
cannot let a real coordinate through, and widening a pattern can.

And binary files are skipped, which is a real limit rather than a safe one. A
coordinate inside a PDF stream or a notebook output would not be seen here. The
2026-08-31 audit checked those by hand -- ten PDFs, thirteen PNGs and their
metadata chunks, a pptx unzipped to its 73 entries -- and found nothing. If
binaries start carrying generated text again, this needs extending.

    python scripts/check_no_coordinates.py [path ...]
"""
import os
import re
import subprocess
import sys

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# What a recorder writes when it never got a fix. Not a position: the absence
# of one, and it is in files that legitimately ship.
NO_FIX = "+000.0000+000.0000"

PATTERNS = {
    "coordinate": r"[+-]\d+\.\d+[+-]\d+\.\d+",
    "api key or token": r"(gh[pousr]_[A-Za-z0-9]{16,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})",
    "private key block": r"BEGIN (RSA |OPENSSH |EC |DSA )?PRIVATE KEY",
}
FATAL = set(PATTERNS)

SKIP_EXT = {"pdf", "png", "jpg", "jpeg", "npy", "h5", "zip", "docx", "wav",
            "pyc", "ico", "gz", "pptx", "xlsx"}


def main():
    targets = sys.argv[1:]
    if not targets:
        targets = subprocess.run(["git", "-C", REPO, "ls-files"],
                                 capture_output=True, text=True).stdout.split()
    hits = {k: [] for k in PATTERNS}
    scanned = skipped = 0
    for f in targets:
        if f.rsplit(".", 1)[-1].lower() in SKIP_EXT:
            skipped += 1
            continue
        p = f if os.path.isabs(f) else os.path.join(REPO, f)
        try:
            text = open(p, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        scanned += 1
        # The sentinel is removed before matching so it cannot mask a real
        # coordinate sitting next to it in the same field.
        clean = text.replace(NO_FIX, "")
        for name, pat in PATTERNS.items():
            m = re.search(pat, clean if name == "coordinate" else text)
            if m:
                hits[name].append((f, m.group(0)[:40]))

    print(f"  tracked text files scanned: {scanned} ({skipped} binaries skipped)")
    bad = 0
    for name in PATTERNS:
        found = hits[name]
        if not found:
            print(f"  {name:20s} clean")
            continue
        bad += name in FATAL
        print(f"  {name:20s} {len(found)} file(s)")
        for f, sample in found[:5]:
            # The sample is truncated and printed because a hit has to be
            # actionable, and a coordinate that is already in a tracked file is
            # already exposed -- printing it in a log changes nothing.
            print(f"      {f}  ::  {sample}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
