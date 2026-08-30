# What is left before this goes to MethodsX

Rewritten 2026-08-26, after the evaluation-contamination fix and the
platform-path fix. Each item says what would make it done, because "done"
has been claimed too early on this project more than once.

## Settled, and how that is known

**The dataset.** 22,169 packed rows from 15,747 clips. The negative class
contains no machine-guessed label: 7,003 clips a person adjudicated one by
one, 1,056 the species expert selected by species, 1,951 curated non-target
recordings, 155 still awaiting a listener.
`check_no_target_in_negatives.py` reads 2,565 human verdicts from four
independent files and finds no target call sitting in a trainable negative.

**The evaluation.** One reviewed detection is scored once. Until 2026-08-25 it
was not: 23 clips relabelled *C. pogonias* were augmented sixteen-fold and
every copy entered the evaluation and threshold-fitting pools, so the sweep
scored 6,478 rows where the review holds 6,110. At IPA2ST, 64 of 144 rows were
copies of four sounds. Fixed, and the corrected sixteen-fold sweep is
`loso16_freqpos_evalfix.csv`.

**The manuscript.** Table 2 and every number tied to it stand on that
corrected sweep. `verify_manuscript_numbers.py` reads 166 OK, 0 OFF, 0 SKIP,
and the same 166 pass from the paper repository against the code repository
next door.

**Colab.** Verified on the machine rather than by simulation, 2026-08-26:
`detection times: 6478/6189 reviewed rows matched [review_gate_table.csv]`,
printed by a fresh clone on a T4. The same line read 0 before the fix. Seven
folds already in Drive were correctly identified as unusable and redone.

## Experiments

1. **`block4`, sixteen folds: DONE 2026-08-27, and the answer is a null.**
   All sixteen folds archived locally from Drive
   (`unfreeze_2026-08-21_drive/block4_loso16.csv`). Frozen macro 0.9604,
   block4 macro 0.9654; paired mean +0.0050 at t = +1.74, better at 10 of 16;
   without IPA4ST +0.0039 at t = +1.37. What the 2026-08-28 audit then
   established, twice-corrected and verified by hand:
   - The training is **unseeded by construction**: `--seed` reaches only the
     train/val split; weight init and batch shuffling use unseeded global
     RNGs. Do not attribute rerun gaps to hardware; the honest word is
     training noise. (Curious but unclaimable: all three CPU reruns beat
     their GPU twins, +0.0056/+0.0230/+0.0072 -- 3 of 3 one direction is
     p = 0.25 under a fair coin. Note it, claim nothing.)
   - Three stations were genuinely trained twice, and the largest rerun gap,
     **0.0230 at IPA18ST, is 4.6x the mean fine-tuning effect**; that
     station's sign against frozen flips with the rerun.
   - **The experiment's overall inference is not robust to run selection**:
     the archived runs give t = +1.74 (p = 0.10); substituting the three
     equivalent local reruns gives t = +2.44 (p = 0.028); substituting
     IPA18ST alone gives t = +2.28. Two defensible analyses of the same
     experiment land on opposite sides of 0.05.
   The only sentence the manuscript can carry: any effect of releasing
   block4 is smaller than the run-to-run variability of the unseeded
   training procedure, and no significance claim in either direction
   survives the choice of which equivalent run is analysed. IPA20ST's
   0.0000 gap is the same file twice, not a replication. The three-fold
   0.6992-to-0.9416 story is an artifact of one station's threshold
   placement and comes out.
   **Done 2026-08-29:** written into the manuscript as the sixteen-fold
   sweep paragraph (limitations, fourth item), once all four arms were in
   clean.

2. **`block34`, sixteen folds: DONE 2026-08-28.** Macro 0.9713 / recall
   0.9217; paired vs frozen +0.0109 at t = +2.45 (without IPA4ST t = +2.28),
   recall unchanged (t = +0.16). Nominally significant -- but it is a single
   unseeded run, and block4's replicates showed run selection alone moves a
   paired mean by ~0.002-0.003. The gap between block34 (+0.0109) and block4
   (+0.0050) is ~2x that scale: suggestive, not provable from one run each.

3. **`nopogonias`: leak-free rerun DONE 2026-08-29; the 2026-08-28 run is
   retired.** The audit had found the first run training on 23 of its own
   scored windows (plus 368 augmented copies) at eight stations:
   --drop-pogonias relabelled the class to Background before the fold masks,
   and keep-all-background exempts Background from station withholding.
   Fixed (relabelled rows keep their station wall; --drop-colobus had the
   same latent hole), verified leak-free at all affected stations, rerun on
   cached features: data/outputs/v13_runs/nopogonias_fixed_2026-08-29/.
   Clean result: macro precision 0.9705 (+0.0102 vs frozen, t = +1.92 --
   no longer nominally past 2), calls kept 87.4% (-0.0462, t = -2.83).
   Eval pools verified identical to frozen at all sixteen stations. The
   leak had inflated macro precision by +0.0034 (per-station up to +-0.019)
   and hidden two thirds of the recall cost (-0.0165 leaky vs -0.0462
   clean). The trade is sharper than the leaky run showed: dropping the
   class buys precision it cannot statistically defend by dropping one
   kept call in twenty, the only effect in the sweep larger than the
   training-noise floor.
   **The noise floor is now MEASURED, not estimated (2026-08-29).** The
   frozen spec was trained twice; only one draw had been scored under the
   corrected masks. Scoring the other through a harness first validated to
   reproduce the published run 16/16 across 12 columns gives, for one
   specification against itself: precision paired mean +0.0035 (t = +1.34),
   SD 0.0103, max single station 0.0256; calls kept -0.0039 (t = -0.33),
   SD 0.046, max single station 0.125. A null comparison reached t = +1.34
   on its own. File: full_2026-08-19/loso16_freqpos_replicate.csv.
   **The one manuscript sentence all four arms support:** every variant moves
   macro precision by at most ~0.011 on single unseeded runs whose
   MEASURED run-to-run mean-shift is 0.0035 and single-fold shift
   up to 0.026; block4's effect is not separable from that noise (replicated),
   block34's and nopogonias's nominal significance cannot be distinguished
   from run-selection luck (unreplicated); and the only variant that moves
   recall, nopogonias, moves it down. The scan comparison, not the re-ranking
   table, is where a real difference would have to show.
   **Written into the manuscript 2026-08-29** (sixteen-fold sweep paragraph,
   limitations, fourth item; guarded by verify_manuscript_numbers.py).

3b. **`nocolobus`: RUN FOR THE FIRST TIME 2026-08-29.** `--drop-colobus` had
   never been executed once, on any dataset, in this repository's history --
   confirmed by parsing the trained class composition out of all 47 run.json
   files rather than trusting the flag. Worth running because balanced
   weights give each of the five classes 0.200 of the loss (the Colobus pair
   takes 0.400) while fold_masks scores zero Colobus rows in any fold: every
   guereza clip is library audio with no station. Result: precision +0.0076
   (t = +1.37), calls kept -0.0133 (t = -1.31), eval pool identical to
   frozen 16/16. The measured floor is +0.0035 at t = +1.34 -- the two
   t-statistics agree to two decimals. File:
   data/outputs/v13_runs/nocolobus_2026-08-29/loso16_nocolobus.csv.

3c. **`temporal_freq` on the corrected pool: RUN 2026-08-29.** The head
   ablation's second measurement was still scored on the pre-evalfix pool,
   so the paper printed 0.9554 for the deployed head in that section and
   0.9604 for the same head, same build, same folds, 570 lines earlier.
   Rerun gives freq 0.9634 vs freqpos 0.9604. Two consequences: the old
   "the sign reversed across datasets" claim is WITHDRAWN (the builds now
   agree, -0.0012 then -0.0030), and a sharper reversal replaces it --
   pairing against the frozen replicate instead gives +0.0005, so the sign
   turns on run selection alone, with all three estimates under the floor.

   **Nothing in the sweep remains unrun.** Five arms plus the replicate;
   every runnable ablation flag in train_v13_loso.py has now been exercised
   on the shipped build except --drop-extra-confuser, which was run on the
   2026-08-19 ablation build and does not need repeating (it deletes rows
   rather than relabelling them, so it never touched the keep-all-background
   exemption the 2026-08-29 audit caught).

## Rescans, and why the existing ones cannot be used

The scans that produced the listening material were run at thresholds fitted
on the contaminated calibration pool. Those thresholds moved when the pool was
corrected, and not slightly:

| station | contaminated | corrected | shift |
|---|---|---|---|
| IPA1ST | 0.6026 | 0.9244 | +0.3218 |
| IPA2ST | 0.9262 | 0.8315 | -0.0947 |
| IPA4ST | 0.9142 | 0.8435 | -0.0707 |
| IPA19ST | 0.9560 | 0.9007 | -0.0553 |
| IPA20ST | 0.3067 | 0.2382 | -0.0685 |

4. **Rescan at the corrected thresholds: 3 of 4 done 2026-08-27**
   (IPA2ST 330, IPA4ST 386, IPA19ST 501; IPA20ST overnight). Two findings the
   listening design must carry:
   - **The pogonias classes of different folds contradict each other**:
     IPA2ST went 4 to 24 pogonias under the corrected threshold while IPA4ST
     went 44 to 1, the 44 reassigned to Cernic. A 177-clip congener boundary
     is exactly where fold-to-fold instability should show first. Sample
     pogonias from both stations separately or the disagreement averages away.
   - **IPA19ST returned 49 Colobus candidates** (old scan: 2). 35 on one day,
     28 in the 07:00 hour -- the shape of a dawn chorus. Median low-frequency
     ratio 0.068, i.e. they fail the same screen our one confirmed field roar
     fails, which is either the screen refuted or the candidates refuted, and
     only ears decide. One confirmed roar rewrites the C. guereza section.
   **Done when:** IPA20ST lands and the stratified expert package is built:
   all 49 Colobus, pogonias from both disagreeing stations, Cernic at random.

## The expert's ears

Nothing here can be done without him, and the first item blocks the abstract.

5. **The 150-clip precision sample.** Sent 2026-08-25 in
   `FOR_SANTI_2026-08-25.zip`; 0 of 150 verdicts returned. The abstract's
   92.7 % came from 55 clips sampled along the confidence axis rather than at
   random, and those verdicts were never written down.
   **Done when:** `compute_precision_from_verdicts.py` prints a proportion and
   a Wilson interval from a sheet with no blank rows, and that sentence
   replaces the 92.7 %.

6. **The 117 dawn windows.** In the same package. The paper says these cannot
   be a low-frequency roar; our own confirmed field roar fails the same test,
   so the claim is unsafe either way.
   **Done when:** they are heard, or the sentence and its count come out.

7. **The detections from the rescan.** This is the question the user has asked
   for throughout: is the detector any good in the field. It cannot be
   answered from counts, and the material must come from item 4 rather than
   from the superseded scans.
   **Done when:** the expert has adjudicated a sample from the corrected
   scans, including the Colobus candidates, and the Method validation section
   reports what he found.

8. **The 155 unheard negatives.** Lowest priority.

## Repository and manuscript hygiene

9. ~~Coordinates in the public repository's history~~ **RESOLVED 2026-08-28.**
   The audit showed all nine recorder positions recoverable from pushed
   history (55 commits, two files), and -- decisive -- from GitHub's 28
   immutable `refs/pull/*/head`, which no force-push can touch. So the fix was
   not a rewrite alone: history was scrubbed with `git filter-repo`
   (`+XX.XXXX+XXX.XXXX` placeholders; fresh-clone sweep finds zero matches on
   any branch), the polluted original was renamed to
   `primates-sound-detection-archive` and stays private with its PR refs
   locked inside, and a fresh repository was created **under the original
   name** with only the clean history -- no PR refs, no dangling objects, no
   forks. The manuscript's Data availability URL is therefore correct without
   edits, and Colab's clone URLs are unchanged. Two full `--all` bundle
   backups predate the rewrite. Addendum 2026-08-29: an exhaustive raw-byte
   audit of every object found ONE survivor the blob scrub missed -- a commit
   MESSAGE quoting IPA11ST's coordinate (filter-repo --replace-text does not
   touch messages, and the verification grep read only file contents). Scrubbed
   with --replace-message, force-pushed (0 forks, 0 PR refs, so nothing pinned
   the old SHAs), and re-verified from an anonymous clone across all three
   channels: blobs 0, messages 0, path names 0. Residual risk: the nine
   positions were public from 2026-07-31 to 2026-08-28 in a zero-fork,
   low-traffic repository; nothing can retract that interval, and the paper
   should never cite the archive repo.

10. **Verifier gaps found by adversarial audit, 2026-08-26.** Four of the 166
    checks are provably incapable of failing; one reports OK across a
    2370/2369 mismatch under an undisclosed tolerance; about 116 compare a
    literal in the verifier against a CSV without reading the manuscript at
    all, so the prose could drift on those numbers silently.
    **Done when:** the cannot-fail checks are removed or made able to fail,
    the tolerance is stated where it is used, and the literals that stand for
    manuscript claims are tethered to the sentence they describe.

11. **Two Table 2 cells disagree with the exact counts** at the third decimal
    (IPA10ST 0.911 against 0.912, IPA8ST 98.0 against 98.1). The table
    faithfully reproduces a 4-decimal CSV; the CSV rounds.

12. ~~**The graphical abstract contradicts the paper.**~~ **RESOLVED
    2026-08-29.** It printed "98.12 % validation accuracy" and "zero
    confusion between the two primate species", both withdrawn in the text;
    "four classes" where five ship; "7x augmentation" where the packer
    cycles four operations to a 3,000-row target; and it listed three
    cleanup filters where two ship disabled. Regenerated from
    make_graphical_abstract.py with the abstract's own validated figures
    (16-fold LOSO over 6,110 reviewed detections, precision 0.71 -> 0.96).
    Figure 1 was found to have the same problem in the same audit and was
    regenerated too: it printed 10-30 % crops against a shipped CHOP_RANGE
    of (0.05, 0.10), +-20 mel bins against TRANSLATE_RANGE (-9, 9), and a
    fixed 7x multiplier from the legacy src/augmentation.py path that the
    v13 packer never calls. Both PDFs had been byte-identical to their
    2026-07-31 renders. NOTE: no automated check covers either file -- the
    verifier's 223 checks read one .tex and nothing else, so a future edit
    to these figures will not be caught.

12b. ~~**The OOD override's stated justification.**~~ **RETRACTED 2026-08-30.**
    The manuscript told a reproducer to copy the 97th-percentile Colobus
    override and justified it by "all nine field-verified roars pass". True of
    the statistics it was fitted on (2026-08-10, IPA4ST head, Colobus p90
    202.9 / p97 328.4); false of every statistics file the repo ships.
    ood_stats/fold_IPA4ST.npz was rewritten under its own filename on
    2026-08-20 and now gives p90 283.7 / p97 377.8, admitting ONE of the nine
    at either percentile -- on the head the override exists for, it changes
    nothing. Best across all five shipped files is 5 of 9. Retracted in
    methodsx_manuscript.tex and src/config.py rather than re-tuned: refitting
    a cutoff on nine clips until those nine pass is choosing a parameter by
    its answer. scripts/score_colobus_controls.py now makes the measurement a
    repo artifact. **Root cause worth keeping in view:** nothing in this
    repository scored the controls between 2026-08-11 and 2026-08-30, so a
    false sentence about the one parameter the paper asks a reader to copy
    stood for nineteen days. The verifier reads one .tex and CSVs; it had no
    way to see this.

13. **MethodsX house style.** Bracketed placeholders remain: co-author name,
    second affiliation, the ethics permit sentence, funding, CRediT.
    The template's own checklist has not been walked end to end.

## The one thing that decides the date

Item 5. Everything else is either an afternoon, a machine running unattended,
or a decision we can make ourselves. The precision figure is in the abstract,
it is the number a reader will quote, and only one person can produce it.
