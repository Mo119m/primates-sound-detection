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
- [x] Data deposit decision — **settled: data stays private** (model weights and
      reference clips available from the corresponding author on request). Both
      placeholders removed: Specifications table and Data availability section now
      state this consistently. No DOI needed.
- [ ] **Santi placeholders** (bold `[Santi ...]`): Background conservation/IUCN
      paragraph + Abstract conservation-significance framing
- [x] **Time-of-day filter** — **settled: not used, not mentioned.** The filter
      description was removed entirely from §1 and the orphaned Limitations bullet
      clause was cleaned up. (The option still exists in the code, just not in the
      paper.)
- [ ] Per-station detection counts (Method validation, bold `[TODO ...]` in
      Field deployment) — pending the full run
- [ ] Citation for the putty-nosed call types (hack/kek/pyow) — get the
      published reference from the species expert and add it (Background, bold
      `[cite the putty-nosed call-type source ...]`)

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
- [x] **Generative-AI use** disclosure — section added after Data availability
      (official template structure), stating an AI assistant helped tidy the
      code repository and edit the documentation/manuscript, with the authors
      taking full responsibility. No tool version required for this scope.
- [ ] For the review manuscript, switch `\documentclass[final,3p,times]` →
      `[review,3p,times]` (double-spaced, line-numbered) if the journal asks.
- [ ] Delete the comment header (lines marked "TODO before submitting").
- [ ] Delete the "Supplementary material [OPTIONAL]" section if unused (or fill
      it in).
- [ ] Compile twice with pdfLaTeX so `\cite` references resolve.

## Decisions already settled (for reference)
- Title, abstract (197 words), keywords (7), Specifications table — done.
- Graphical abstract: separate file only, not embedded in the body — done.
- Augmentation figure kept as one (A)/(B) multi-panel figure — intended.
- Method details §1–§8 verified line-by-line against the code.
