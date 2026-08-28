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
   without IPA4ST +0.0039 at t = +1.37. The stronger sentence than any t,
   corrected by the 2026-08-28 audit: three stations were genuinely trained
   twice (once per machine, same seed, different hardware), and the two runs
   differ by 0.0056 (IPA17ST), 0.0230 (IPA18ST) and 0.0072 (IPA19ST).
   **The largest run-to-run gap, 0.0230, is 4.6 times the mean effect of
   fine-tuning**, and at IPA18ST the sign against frozen flips with the rerun
   (-0.0076 archived, +0.0154 local). Quote the largest gap, not a middling
   one, and note IPA20ST's 0.0000 is the same file twice, not a replication.
   The manuscript's three-fold 0.6992-to-0.9416 story is an artifact of one
   station's threshold placement and comes out.
   **Remaining:** write this into the manuscript once block34 lands, so the
   section is rewritten once, not twice.

2. **`block34`, sixteen folds: 7 of 16**, running on Colab in ~2 h sessions
   (the free tier reclaims the runtime; each disconnect costs only the fold in
   flight and a click on the Drive consent when resumed).

3. **`nopogonias`, sixteen folds.** Not started. Frozen trunk, *C. pogonias*
   dropped. The paper reports a null on this, but that null was last measured
   on 8 August against a dataset with no field pogonias in it, so it cannot
   speak to the 27 field clips the class now carries. Minutes per fold, not
   half an hour: it trains on cached features.
   **Done when:** the limitations section states the result measured on the
   dataset the claim is about, rather than one that predates it.

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

9. **`primates-sound-detection` is public and its history carries recorder
   coordinates** -- 304 instances across nine positions, for a primate that is
   hunted. `primates-paper` is private and clean. The manuscript's Data
   availability section points at the public one, so making it private breaks
   the link a reviewer follows.
   **Done when:** either a coordinate-clean public mirror exists and the
   manuscript points at it, or the availability statement is rewritten around
   what will actually be public.

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
