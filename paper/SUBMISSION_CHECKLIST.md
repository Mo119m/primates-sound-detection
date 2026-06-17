# Pre-submission checklist — MethodsX

Things to finish **before** submitting `methodsx_manuscript.tex`. Tick them off
as they are done. (This file is a working note, not part of the manuscript.)

## Content placeholders to fill (search the .tex for `[` and `TODO`)
- [ ] Co-author name (`[Co-author Name]`) — author list **and** CRediT section
- [ ] Second affiliation `[Second institution, full address]` / `[Country]`
      (delete affiliation `[b]` entirely if the co-author is also at UW–Madison)
- [ ] CRediT roles for the co-author
- [ ] Ethics statement: permit `[number]` / `[authority]` — or delete that
      bracket if no permit applies (passive recording)
- [ ] Acknowledgments: field teams / funding (or keep the "no funding" line)
- [ ] Resource availability (Specifications table): data DOI/repository, or
      remove the `[Add a data DOI...]` bracket if data stays private
- [ ] **Santi placeholders** (bold `[Santi ...]`): Background conservation/IUCN
      paragraph + Abstract conservation-significance framing
- [ ] **Time-of-day filter** (`§1`, `[TODO]`): confirm whether the final
      detection run used the 05:30–10:30 filter, and reconcile the wording
- [ ] Per-station detection counts in Method validation (pending local run)

## Figures / artwork (MethodsX requirements)
- [ ] **Rename figures for separate upload**: `Figure_1.pdf … Figure_4.pdf`
      (order of appearance: augmentation, model architecture, training curves,
      confusion matrix). The in-`.tex` `\includegraphics` names can stay; only
      the uploaded files need the Figure_N convention.
- [ ] Upload `graphical_abstract.pdf` as the **separate graphical-abstract**
      file in the submission system (it is intentionally not in the body).
- [x] Figures are vector PDF with **TrueType** fonts embedded (no Type 3).
- [ ] Upload the figure PDFs to Overleaf/submission (they were regenerated;
      re-upload the latest versions).

## Formatting / declarations
- [ ] Add the **Generative-AI use** disclosure statement (new section before
      References) — required because AI tools assisted manuscript/figure/code
      preparation. See the MethodsX Guide ("Declaration of generative AI use").
- [ ] For the review manuscript, switch `\documentclass[final,3p,times]` →
      `[review,3p,times]` (double-spaced, line-numbered) if the journal asks.
- [ ] Delete the comment header (lines marked "TODO before submitting").
- [ ] Compile twice with pdfLaTeX so `\cite` references resolve.

## Decisions already settled (for reference)
- Title, abstract (197 words), keywords (7), Specifications table — done.
- Graphical abstract: separate file only, not embedded in the body — done.
- Augmentation figure kept as one (A)/(B) multi-panel figure — intended.
- Method details §1–§8 verified line-by-line against the code.
