"""Every committed notebook must be one a person could actually run.

colab/v13_train.ipynb was unrunnable from the day it was generated on
2026-07-31 until 2026-08-31 -- ten of its eleven code cells were a single line
each and five were syntax errors, because its generator built cell sources with
split("\\n"), which strips the newlines a notebook source list has to carry.
Nothing noticed for a month. The notebook trains the paper's headline result and
sits in the repository the manuscript points a reader at.

Reading the generator would not have caught it; the bug is only visible in the
generated file. So this reads the generated files, and it is in tests/ rather
than in a script because tests/ is what CI already runs.
"""
import ast
import glob
import json
import os

import pytest

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
NOTEBOOKS = sorted(glob.glob(os.path.join(REPO, "**", "*.ipynb"), recursive=True))


def _cells(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["cells"]


def test_there_are_notebooks_to_check():
    """A glob that silently matches nothing is not a passing test."""
    assert NOTEBOOKS, "no notebooks found; this test would pass vacuously"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: os.path.basename(p))
def test_source_lines_keep_their_newlines(path):
    """The failure that hid for a month: statements fused into one line.

    A notebook source list is concatenated with no separator when the cell
    runs, so every element except the last has to end in a newline. If none of
    them does and there is more than one element, the cell has been flattened.
    """
    for i, cell in enumerate(_cells(path)):
        src = cell.get("source")
        if not isinstance(src, list) or len(src) < 2:
            continue
        assert any(line.endswith("\n") for line in src[:-1]), (
            f"{os.path.basename(path)} cell {i}: source list carries no "
            f"newlines, so the cell will run as one line. Its generator "
            f"probably used split(chr(10)) where it needed "
            f"splitlines(keepends=True). First element: {src[0][:60]!r}"
        )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: os.path.basename(p))
def test_code_cells_parse(path):
    """Python cells must be valid Python once the shell lines are removed.

    Lines beginning with ! or % are IPython, not Python, and a backslash at the
    end of one continues it -- so the continuation lines go too, or what is
    left looks like a stray indent. Getting that wrong produced four false
    alarms against a notebook that was fine, which is worth more care than the
    check itself.
    """
    for i, cell in enumerate(_cells(path)):
        if cell.get("cell_type") != "code":
            continue
        keep, skipping = [], False
        for line in "".join(cell["source"]).splitlines():
            stripped = line.lstrip()
            if skipping or stripped.startswith(("!", "%")):
                skipping = line.rstrip().endswith("\\")
                continue
            keep.append(line)
        body = "\n".join(keep)
        if not body.strip():
            continue
        try:
            ast.parse(body)
        except SyntaxError as error:
            pytest.fail(
                f"{os.path.basename(path)} cell {i} is not valid Python: "
                f"{error.msg} at line {error.lineno}\n  {body[:200]!r}")
