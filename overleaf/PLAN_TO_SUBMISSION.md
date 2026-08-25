# The road to MethodsX, in order

Rewritten 2026-08-25 after Colab Pro arrived. Supersedes every earlier version.
One list, ordered by what blocks what. An item is DONE only when its checks
pass, and "clean" below always names the check that says so.

## Where the clean claims actually stand

| claim | status | the check that says so |
|---|---|---|
| data | clean | check_no_target_in_negatives.py: 2,565 human verdicts, 4 sources, zero target calls in trainable negatives; pointed at the current 22,169-row index as of today |
| manuscript numbers | clean | verify_manuscript_numbers.py: 157 OK, 0 OFF, 0 SKIP |
| manuscript structure | clean | check_manuscript_latex.py, check_manuscript_labels.py: PASS |
| assembled models | clean | seam + permutation checks inside assemble_fold_model.py, recorded for all 5 scan models |
| methods text vs code | **being audited now** | 12-agent adversarial audit running 2026-08-25; results pending |
| fine-tuning experiment | **1/48 folds** | gated-column guard now rejects unusable folds on both machines |

History that justifies the caution: on 2026-08-24 "Colab is running fine"
was said while it wrote three folds missing every gated column, and six prose
numbers sat two runs stale while 147 checks passed around them. A claim of
clean now comes with the name of its check or it is not a claim.

## 1. The compute, all of it on Colab Pro now

Open `colab/v13_unfreeze.ipynb` fresh from GitHub -- the title must say
**edited 2026-08-25** -- and Run All. Everything it needs is in the clone;
nothing gets uploaded. It refuses to train if the review gate table is missing
rather than writing 48 more unusable folds.

| arm | folds | cost/fold | what it settles |
|---|---|---|---|
| block4 | 16 | ~25 min GPU | is releasing one VGG19 block worth it |
| block34 | 16 | ~25 min GPU | are two blocks worth it |
| nopogonias | 16 | minutes | the paper's null on dropping the class, currently measured on a dataset that predates every field pogonias clip |

Local machine works the same arms backwards from IPA20ST as a slow second
engine; whichever side reaches a station first does it. Colab Pro should carry
nearly all of it in one long session (~14 GPU-hours).

When the folds are in: rewrite the fine-tuning section, which currently rests
on a 3-fold run whose entire spread came from one station with a 4 % base rate.

## 2. What only Santi can do, in priority order

| batch | size | blocks |
|---|---|---|
| precision sample (`FOR_SANTI_2026-08-25.zip`, on Desktop) | 150 | **the abstract's precision figure -- the submission date itself** |
| dawn windows | 117 | the "cannot be a roar" sentence stays or goes |
| new-scan clips (12 Colobus + 68 pogonias + 90 Cernic + 22 birds) | 185 | whether the retrained scanner's extra detections are calls or errors; one confirmed Colobus roar rewrites that branch |
| leftover negatives | 155 | nothing in the paper; dataset hygiene only |

The zip is 31.9 MB -- too big for a Gmail attachment, send as a Drive link.

## 3. Detections still owed, and what they wait on

- **Scans of the fine-tuned arms.** The manuscript claims fine-tuning
  "redistributes false positives rather than reducing them" from the OLD
  3-fold scans. After item 1, scan block4/block34 at IPA1ST, IPA2ST, IPA4ST at
  fitted thresholds and either re-support or cut that sentence. Caveat to
  resolve first: a fine-tuned fold's weights include the unfrozen trunk layers
  (132 MB vs 23 MB), so verify assemble_fold_model.py restores them before
  trusting any scan.
- **OOD statistics for the scan models.** The three existing fitted scans ran
  un-gated because no stats were fitted for those heads; the counts are upper
  bounds relative to the described pipeline. Build stats, rerun the three
  scans, and the numbers become the pipeline's own. (build_ood_stats.py, one
  command per model.)
- **Nothing else.** Thirteen unscanned stations stay unscanned; the paper
  never promises them.

## 4. Writing, after the above lands

1. Fine-tuning section on 48 folds (after item 1).
2. Precision paragraph + Wilson interval from the 150 (after Santi).
3. Dawn-window sentence: keep, weaken, or delete (after the 117).
4. drop_pogonias null: replace the "predates the field clips" caveat with the
   remeasured answer (after item 1).
5. Whatever the methods-vs-code audit returns (pending today).
6. Placeholders: Santi's affiliation, corresponding-author switch, ethics
   permit line, funding, CRediT, acknowledgments.
7. Final gate, in this order: regen_table2.py --write, then
   verify_manuscript_numbers.py reads 157+ OK 0 OFF 0 SKIP, latex + labels
   PASS, coordinate scan of every file leaving the machine.

## The critical path

Santi's 150 is the only item nobody else can start, finish, or hurry. Colab
Pro makes item 1 an afternoon. Everything in 3 and 4 is hours once its
dependency lands. Send the zip first.
