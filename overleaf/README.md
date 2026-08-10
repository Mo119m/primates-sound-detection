# Overleaf folder

Everything the MethodsX manuscript needs to compile, and nothing else. Overleaf
builds this project from the repository, so a file that lives elsewhere is a
build failure rather than a local inconvenience.

**Main document:** `methodsx_manuscript.tex`

| File | Role |
|---|---|
| `methodsx_manuscript.tex` | The manuscript. `elsarticle` class, which Overleaf ships. |
| `Figure_1.pdf` | Pipeline overview |
| `Figure_2.pdf` | Model architecture |
| `Figure_3.pdf` | Augmentation |
| `graphical_abstract.pdf` | **Not compiled.** MethodsX requires it as a separate upload at submission; it is kept here so the submission set is in one place. |

No `.bib` file: the 13 references are inline `\bibitem` entries, so nothing needs
BibTeX. Packages used are `amssymb`, `booktabs`, `float`, `fontenc`, `graphicx`,
`inputenc`, `lineno`, `hyperref`, `tabularx` — all standard on Overleaf.

## Before submitting

Six bracketed placeholders and two marked for the co-author are still open. They
are the only things in the manuscript that cannot be filled from the data:

- co-author name (author list **and** the CRediT section)
- second affiliation and country — delete affiliation `[b]` entirely if the
  co-author is also at UW–Madison
- the two `[Santi ...]` passages: the conservation framing in the abstract and
  the background paragraph
- ethics: permit number and issuing authority, or delete that sentence if
  passive recording required no permit

`scripts/../scratchpad/check_overleaf_folder.py` verifies the folder is
self-contained and counts what is left open.

## A note on visibility

This repository is public, and Overleaf syncing from it means the manuscript is
publicly readable before peer review. That is a publishing decision rather than
a technical one, and it affects a co-author as much as the corresponding author.
Making the repository private, or moving the manuscript to a private repository
of its own and syncing Overleaf with that, both leave the code public — which is
worth keeping, since a methods paper is judged partly on whether its pipeline
can be run.
