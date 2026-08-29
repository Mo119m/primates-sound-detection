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
   **Remaining:** write this into the manuscript once block34 lands, so the
   section is rewritten once, not twice.

2. **`block34`, sixteen folds: DONE 2026-08-28.** Macro 0.9713 / recall
   0.9217; paired vs frozen +0.0109 at t = +2.45 (without IPA4ST t = +2.28),
   recall unchanged (t = +0.16). Nominally significant -- but it is a single
   unseeded run, and block4's replicates showed run selection alone moves a
   paired mean by ~0.002-0.003. The gap between block34 (+0.0109) and block4
   (+0.0050) is ~2x that scale: suggestive, not provable from one run each.

3. **`nopogonias`: the 2026-08-28 run is INVALID and is being redone.**
   The 2026-08-29 audit found it training on 23 of its own scored windows
   (plus 368 augmented copies) at eight stations: --drop-pogonias relabelled
   the class to Background before the fold masks, and keep-all-background
   exempts Background from station withholding. Its +0.0135 precision points
   exactly where memorising your own evaluation false positives points.
   Fixed (relabelled rows keep their station wall; --drop-colobus had the
   same latent hole), verified leak-free at all affected stations, rerunning
   on cached features. The trade description below is provisional until the
   clean run lands.
   ~~DONE 2026-08-28, and the null is now a trade.~~ Macro precision 0.9739 (+0.0135 vs frozen, t = +2.40) bought with
   recall 0.9037 (-0.0165, t = -1.45). Dropping the class no longer "costs
   little either way": it buys precision by dropping calls. The limitations
   section must state this measured-on-the-right-dataset result and retire
   the 8 August null.
   **The one manuscript sentence all four arms support:** every variant moves
   macro precision by at most ~0.014 on single unseeded runs whose
   demonstrated run-to-run mean-shift is ~0.002-0.003 and single-fold shift
   up to 0.023; block4's effect is not separable from that noise (replicated),
   block34's and nopogonias's nominal significance cannot be distinguished
   from run-selection luck (unreplicated); and the only variant that moves
   recall, nopogonias, moves it down. The scan comparison, not the re-ranking
   table, is where a real difference would have to show.

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

12. **The graphical abstract contradicts the paper.** It prints "98.12 %
    validation accuracy", which the manuscript has a section explaining it
    will not report; it says four classes where there are five; it says
    7x augmentation where the scheme is a target count.

13. **MethodsX house style.** Bracketed placeholders remain: co-author name,
    second affiliation, the ethics permit sentence, funding, CRediT.
    The template's own checklist has not been walked end to end.

## The one thing that decides the date

Item 5. Everything else is either an afternoon, a machine running unattended,
or a decision we can make ourselves. The precision figure is in the abstract,
it is the number a reader will quote, and only one person can produce it.
