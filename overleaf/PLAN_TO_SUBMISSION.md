# What is left before this goes to MethodsX

Written 2026-08-25, after the dataset was closed at 22,169 rows.

## Settled

The dataset is final and the negative class is no longer machine-labelled at
all: 15,747 clips, 58.9 % expert-verified per clip, 11.9 % reviewer-verified,
29.2 % provenance-based curated recordings. C_pogonias has 27 field clips where
it had none. `check_no_target_in_negatives.py` reads four independent sources of
human verdicts, 2,565 in total, and finds no target call sitting in a trainable
negative.

The code that built it is committed, `main` now matches the working branch, and
the manuscript has moved to its own repository so it can be shared without the
recorder coordinates that the code repository carries in its history.

## Blocked on the expert, in priority order

Nothing here can be done without him, and the first item blocks the abstract.

1. **The 150-clip precision sample.** Built and blinded as `s0000.wav` to
   `s0149.wav` in `data/outputs/precision_resample/`; 0 of 150 verdicts in.
   The abstract's 92.7 % came from 55 clips sampled along the confidence axis
   rather than at random, 56 % inclusion in the top stratum against 2.0 % below
   it, and the verdicts were never written down. Until these 150 come back the
   paper has no defensible precision figure, and a Wilson interval on the old
   sample is invalid for the same reason.

2. **The 117 dawn windows.** The manuscript currently says they cannot be a
   low-frequency roar. Our own confirmed field roar has 98.0 % of its in-band
   energy above 1.5 kHz, so the screen would have discarded it too. Either these
   are heard, or the sentence and the count come out.

3. **The 185 clips from the retrained scans** -- 12 Colobus candidates, 68
   pogonias, 90 Cernic, 22 birds. These decide whether the retrained head's
   extra detections are calls the deployed model missed or false positives it
   avoided. The counts alone cannot say, and the 12 Colobus matter out of
   proportion to their number: the paper reports that branch as a negative
   result because no field roar has been confirmed here, and one confirmed roar
   changes both that and the denoising arm.

4. **The 155 still-unheard negatives.** The smallest job and the least urgent.

## Compute we still owe

5. **Table 2 on the current dataset.** No compute needed -- 
   `full_2026-08-19/loso16_freqpos.csv` has all sixteen folds with every
   `gated_` column filled. The table in the paper is the 08-18 run at 21,120
   training rows. Swapping it moves precision 0.9695 to 0.9554 and recall
   0.9071 to 0.9163. Do this first; it is an afternoon.

6. **The sixteen-fold fine-tuning comparison.** 3 of 32 folds done, all on
   Colab. This is not optional: the manuscript already reports a fine-tuning
   result from a three-fold run whose entire spread came from IPA4ST, where 100
   calls in 2,470 detections make precision a knife edge a fitted threshold
   lands either side of. The local route is dead -- 50 CPU-hours produced zero
   folds, because unfreezing reads the image pack rather than the feature cache
   and 10.23M trainable parameters will not backpropagate through a CPU at any
   useful rate. Colab does a fold in 25 minutes, so 32 folds is about 13 GPU-
   hours, which on a free quota is several days of starting the notebook.
   The fallback, if the quota will not stretch: cut the claim back to what three
   folds support and say plainly that it is three folds and that one station
   carries it.

7. **Scans beyond the three stations.** IPA1ST, IPA2ST and IPA4ST are scanned.
   The paper does not need the other thirteen and should not wait for them.

## Writing

8. Every number that moved with the dataset, not just Table 2.
   `verify_manuscript_numbers.py` is the instrument: 147 checks, and it must
   read 147 OK, 0 OFF, 0 SKIP before submission. A SKIP is not a pass.
9. The bracketed placeholders: co-author name, second affiliation, ethics
   permit sentence, funding, CRediT, acknowledgments.
10. Authorship. Currently the manuscript has Moshi Fu as corresponding author.
    The intent is for the expert to take last author and corresponding author.

## The one thing that decides the date

Item 1. Everything else is either an afternoon of work or a decision we can
make ourselves. The precision figure is in the abstract, it is the number a
reader will quote, and only one person can produce it.
