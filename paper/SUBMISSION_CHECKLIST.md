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
- [x] Per-station detection counts (Field deployment) — filled from the manual
      review of all 16 stations: 6189 *C. nictitans* detections, 2535 confirmed,
      41.0% precision, with the per-station table (`tab:field`). Regenerate with
      `python scripts/summarize_review.py --dir reviews/`.
- [x] Cleanup effect (Field deployment) — rewritten around leave-one-station-out
      results. Filters: 41.0%→46.6% over 16 stations, 65.5%→83.6% over the 15
      excluding IPA4ST (91.3% of calls kept, 66.0% of false positives removed).
      Review ordering: mean average precision 0.900 vs 0.658 for arbitrary
      order, above baseline at 15/15 stations, recovering 90.3% of recoverable
      calls from half the clips with no fitted parameter (`tab:ranking`).
      Negative results reported: YAMNet (disabled, `config.USE_YAMNET_FILTER`)
      and in-sample inflation (61.1% → 38.4% held out). IPA4ST documented as a
      failure mode. Regenerate with
      `python scripts/evaluate_cleanup.py --review reviews/ --cleanup <dir>`
- [ ] *Colobus guereza* field verification — detections exist but are not yet
      manually reviewed. Either add them, or keep the current sentence scoping
      the field results to *C. nictitans*.
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

- [x] **Method details now describe what the validation reports.** Added
      "Organising the review: episodes and ordering" to §Method details: (i) the
      five-minute episode grouping rule and (ii) the averaged within-station
      percentile-rank ordering over four signals. Previously these appeared only
      in Method validation, which left the paper's headline results without a
      method to point at.
- [x] Removed the stale claim (Field deployment) that IPA4ST "is precisely the
      failure mode the automatic cleanup is designed to absorb" — it is the
      opposite, and the sentence also referred to the disabled audio tagger.

- [x] **Call-event counts (Field deployment)** — filled. 6189 windows = 2103
      call events; 2535 confirmed windows = **755 confirmed vocalizations**
      (inflation 3.36). Genuine events span 3.40 windows vs 2.69 for false
      positives; 20.3% of genuine events are single-window vs 61.0% of false
      positives. Event-level precision 35.9% (lower than the 41.0% window-level
      figure, because genuine calls consolidate more). Regenerate with
      `python scripts/evaluate_cleanup.py --review reviews/ --cleanup <dir>`
- [ ] **Field recall (optional, adds a result)** — tooling is in place
      (`scripts/recall_sample.py plan|budget|score`). ~3.5 h of exhaustively
      annotated audio puts recall inside ±5%. Limitations currently states the
      method and says the number is not reported; replace that with the number if
      the listening gets done.
- [x] Ranking-signal search — six temporal features derived from timestamps were
      tested and **rejected**: held-out gain +0.011, bootstrap 95% CI
      [-0.004, +0.027], and two of fifteen stations supply nearly all of it.
      Reported as a negative result, with the positive finding inside it (episode
      size alone reaches 0.827 average precision against 0.845 for detector
      confidence). Reproduce with `python scripts/rank_signals_experiment.py`.

- [ ] **If the model is retrained** — the Limitations paragraph on circular bias
      ("hard negatives are scored by the same model that produced them") should be
      revised: `scripts/mine_hard_negatives.py` mines from the manual review
      instead, which removes the circularity. Also update the field numbers, and
      report the gate's held-out-station result rather than a pooled one.

## Decisions already settled (for reference)
- Title, abstract (197 words), keywords (7), Specifications table — done.
- Graphical abstract: separate file only, not embedded in the body — done.
- Augmentation figure kept as one (A)/(B) multi-panel figure — intended.
- Method details §1–§8 verified line-by-line against the code.
