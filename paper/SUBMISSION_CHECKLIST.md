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
- [ ] **Time-of-day filter** — **REOPENED 2026-07-31. It is now on by default and
      it changes every field number in the paper.** `src/config.py` ships
      `TIME_FILTER_START='05:00'` / `TIME_FILTER_END='19:00'`; the previous
      05:30-10:30 window was never validated and keeps only 40.2 % of the
      confirmed calls. Recalibrated against all 6 189 reviewed detections by
      `scripts/calibrate_time_gate.py`:

          gate off      6 189 detections, 2 535 calls, precision 0.4096
          05:00-19:00   3 571 detections, 2 503 calls, precision 0.7009

      Every field figure currently in the manuscript was computed with the gate
      OFF and nothing says so, so a reader reproducing today gets the second row.
      Two things have to happen before this can be ticked again:
        1. State the regime wherever a field number appears, or the validation
           section is not reproducible.
        2. Any diel/time-of-day claim must be measured with the gate OFF. 32 of
           the 2 535 confirmed calls fall outside the window; recovering the
           window you imposed is circular.
- [x] Per-station detection counts (Field deployment) — filled from the manual
      review of all 16 stations: 6189 *C. nictitans* detections, 2535 confirmed,
      41.0% precision, with the per-station table (`tab:field`). Regenerate with
      `python scripts/summarize_review.py --dir reviews/` (or read
      `data/outputs/auto_cleanup/cleanup_vs_review.csv`, which is in the repo and
      carries the same 6 189 verdicts).
- [x] Cleanup effect (Field deployment) — rewritten around leave-one-station-out
      results. Filters: 41.0%→46.6% over 16 stations, 65.5%→83.6% over the 15
      excluding IPA4ST (91.3% of calls kept, 66.0% of false positives removed).
      Review ordering: mean average precision 0.900 vs 0.658 for arbitrary
      order, above baseline at 15/15 stations, recovering 90.3% of recoverable
      calls from half the clips with no fitted parameter (`tab:ranking`).
      Negative results reported: YAMNet (disabled, `config.USE_YAMNET_FILTER`)
      and in-sample inflation (61.1% → 38.4% held out). IPA4ST documented as a
      failure mode. Regenerate with
      `python scripts/evaluate_cleanup.py --review
      data/outputs/auto_cleanup/cleanup_vs_review.csv --cleanup <dir>`
- [x] *Colobus guereza* field verification — **done, and it came back empty.**
      All 253 detections were listened to: **zero genuine roars**, field
      precision 0.0%, median confidence 0.953. Because the field record contains
      no confirmed positive, field recall for this species is not estimable and
      the class contributes no evaluation positive to any of the 16 LOSO folds.
      The manuscript now reports this as a negative result rather than claiming
      a detection (Abstract and Field deployment both rewritten).

      Supporting evidence gathered 2026-08-01, all reproducible:
        * 194 h of dawn audio (05:00–08:00), 390 recordings, all 16 stations,
          698 007 windows scored (`scripts/colobus_dawn_probe.py`). 124 windows
          exceeded 0.10 on the Colobus channel; **117 are physically incapable of
          being a roar** (>95% of energy above 1.5 kHz) and the 7 that survive
          the screen were listened to and are not *C. guereza*.
        * Positive control on the same audio: Cernic fires ≥0.9 in 84 recordings,
          so the null is about Colobus, not about the pipeline.
        * **This is a statement about the detector, not about the forest.** All
          789 training positives are Macaulay Library media recorded elsewhere; a
          model that cannot transfer to this channel would look identical. Say so
          wherever the negative is stated.
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
      `python scripts/evaluate_cleanup.py --review
      data/outputs/auto_cleanup/cleanup_vs_review.csv --cleanup <dir>`
- [x] **Field recall — settled: not reported, and the cost is reported instead.**
      The old note here ("~3.5 h of annotated audio puts recall inside ±5%") was
      wrong by more than an order of magnitude and is what the Limitations
      section used to claim. The target is sparse: 755 confirmed vocalization
      events over 513 recordings = **2.9 events/h in recordings the detector
      fired in at all, 0.50/h over the full 1 506 h corpus.** A Wilson interval
      of half-width h around recall ≈ 0.9 needs ~1.96²p(1−p)/h² observed calls:

          ±5 points → 138 calls → 277 h random sampling, 47 h if restricted
                                   to active recordings
          ±10       →  35 calls →  69 h / 12 h
          ±20       →   9 calls →  17 h /  3 h

      3.5 h buys **±18 points**, not ±5. And the cheaper column is not free:
      sampling only recordings that already contain detections estimates recall
      *conditional on the species being active*, a narrower quantity than field
      recall. Limitations now carries this table. Reopen only if a reviewer
      insists on a number and someone commits ~12 h of listening for ±10 points.
- [x] Ranking-signal search — **numbers corrected 2026-08-02.** It is **nine**
      features, not six (6 episode + 3 event; `episode_features.py`), and the
      held-out gain is **+0.0152**, bootstrap 95% CI **[−0.0012, +0.0353]**, with
      **13 of 15** stations improving and **+0.0045** left after setting aside the
      two largest contributors. The previous values (+0.011, [−0.004, +0.027],
      12/15, 0.002) do not reproduce. Re-run:
      `python scripts/rank_signals_experiment.py --matched
      data/outputs/auto_cleanup/cleanup_vs_review.csv --exclude-station IPA4ST`
      Still a negative result, with two positive findings inside it now stated in
      the manuscript: episode size alone reaches 0.827 average precision against
      0.845 for detector confidence, and **`event_windows` alone reaches 0.9039,
      marginally above the 0.8997 of the four-signal ordering we ship.** That
      second one needs its tie-break caveat to travel with it — `event_windows`
      takes 38 distinct values, its largest tie holds 17.8% of detections, and a
      stable sort inflates it to 0.9098.

## Decisions already settled (for reference)
- Title, abstract (197 words), keywords (7), Specifications table — done.
- Graphical abstract: separate file only, not embedded in the body — done.
- Augmentation figure kept as one (A)/(B) multi-panel figure — intended.
- Method details §1–§8 verified line-by-line against the code.
