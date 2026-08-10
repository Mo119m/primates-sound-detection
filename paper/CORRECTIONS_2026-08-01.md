# CORRECTION DOCUMENT — `paper/methodsx_manuscript.tex`
### Branch `v13-honest-labels` · prepared 2026-08-01 · items are independently approvable

Line numbers below were re-counted directly against the file this session, not taken from summaries. Where an earlier note gave a range off by one or four lines, the range here supersedes it. Every "current" block is a verbatim transcription including line breaks.

Two facts frame everything else:

- **The Colobus branch has a field true-positive count of zero.** 253 detections, all listened to, none genuine, median confidence 0.953.
- **The 98.12 % validation accuracy is a measurement of memorisation.** The split was taken after augmentation; ~79 % of source clips leaked a near-duplicate across it.

Both are currently stated in the paper as successes.

---

# 1. SUBMISSION-BLOCKING

## B1 — Abstract claims a detection that has never occurred (lines 70–74)

**Current (verbatim, lines 70–74):**
```latex
detection must be checked by ear. We detect \textit{Cercopithecus nictitans}
and \textit{Colobus guereza} at Makokou, Gabon. Mel-spectrograms are classified
by a VGG19 transfer-learning model whose frequency-position-aware CRNN head
keeps forest birds and insects from being mistaken for the target, reaching
98\% validation accuracy with zero confusion between the species. A cleanup,
```

**Why it is false:** the deployment returned 253 `Colobus_guereza` detections; the project owner listened to all 253 and none is a genuine roar. `src/config.py:215-216` says so in the repository itself. No *C. guereza* vocalisation has ever been confirmed at this site. Separately, "98 % validation accuracy with zero confusion between the species" is a leaked-split number (see B3), and the "zero confusion" is a separation of *site-recorded Cernic* from *Macaulay-Library Colobus* — a recording-domain result, not a species result.

**Replacement:**
```latex
detection must be checked by ear. We target \textit{Cercopithecus nictitans}
and \textit{Colobus guereza} at Makokou, Gabon; only \textit{C.\ nictitans} was
confirmed at the site, and the \textit{C.\ guereza} branch is reported as a
negative result. Mel-spectrograms are classified by a VGG19 transfer-learning
model whose frequency-position-aware CRNN head keeps forest birds and insects
from being mistaken for the target. Measured by a 16-fold
leave-one-station-out sweep over 6\,189 manually reviewed field detections,
retraining raises precision from 0.70 to 0.93 while retaining 95\,\% of the
calls the deployment found. A cleanup,
```

---

## B2 — The low-frequency gate passage is inverted, and its central claim is falsified by the deployment's own CSVs

This is two edits in two places. Both must land together or the paper contradicts itself.

### B2a — Detection section (lines 455–466)

**Current (verbatim, lines 455–466):**
```latex
below 1.5\,kHz is measured. Genuine \textit{Colobus} roars concentrate almost
all of their energy in the low band and sit well above the gate threshold,
whereas the high-frequency bird and insect sounds that the classifier
occasionally mis-fires on carry almost all of their energy above 1.5\,kHz.
Detections whose low-frequency ratio falls below the threshold (0.20) are
dropped from the detection list (i.e.\ treated as non-target Background). The
threshold was calibrated jointly on the curated \textit{Colobus} reference
clips and the field false positives: it sits above the highest low-frequency
ratio observed among the insect and bird false positives ($\sim$0.09) and below
the fifth percentile of the reference clips ($\sim$0.26), so it removes
essentially all of these false positives while retaining roughly 98\% of
genuine reference clips.
```

**Why it is false — three separate errors.** (1) The committed threshold is **0.40**, not 0.20 (`src/config.py:249`). (2) "calibrated jointly on … the field false positives" is untrue: the ~0.09 maximum is a property of the `Colobus_confuser` *training* clips (`src/config.py:211-214`), which the field never resembled. (3) The field false positives are **not** high-frequency. All 253 Colobus rows in `data/outputs/detections/**/*.csv` have `low_freq_ratio` ≥ 0.2007, median 0.42, p75 0.84, max 0.98. Every one of them cleared a 0.20 gate. The gate as deployed removed **none** of them, because the dominant field confuser is thunder, which is itself low-frequency.

**Replacement (lines 455–466):**
```latex
below 1.5\,kHz is measured. Genuine \textit{Colobus} roars concentrate almost
all of their energy in the low band and sit well above the gate threshold, as
do the archival reference clips; the high-frequency bird and insect sounds that
earlier model versions mis-fired on carry almost all of their energy above
1.5\,kHz. Detections whose low-frequency ratio falls below the threshold are
dropped from the detection list (i.e.\ treated as non-target Background).

The deployment reported here used a threshold of 0.20. That value was
calibrated on the curated \textit{Colobus} reference clips and on the
\textit{Colobus\_confuser} training clips, whose low-frequency ratios do not
exceed 0.092, so 0.20 sat in a wide empty gap and retained 97.6\,\% of the
reference clips. The calibration did not survive contact with the field. The
253 \textit{Colobus} detections the deployment returned---every one of which
was subsequently confirmed by ear to be spurious---have low-frequency ratios
spanning 0.20 to 0.98, with a median near 0.40, because the dominant field
false positive is thunder, which is itself low-frequency. The gate at 0.20
therefore removed none of them. The repository now sets the threshold to 0.40
(\texttt{LOWFREQ\_GATE\_THRESHOLD}, \texttt{src/config.py}), which retains
93.4\,\% of reference clips and removes roughly half of the 253. The gate
should be read as a partial mitigation against one high-frequency confuser,
not as a control on the \textit{Colobus} false positives actually observed.
```

### B2b — Development-history section (lines 1110–1117)

**Current (verbatim, lines 1110–1117):**
```latex
calibrated low-frequency energy gate at detection time (threshold 0.20;
see \S\,Sliding-window detection) that catches residual high-frequency false
positives whose energy lies almost entirely above 1.5\,kHz. An earlier version
of the gate with a higher threshold (0.40) was too aggressive and flagged
genuine \textit{Colobus} clips whose ambient insect energy lowered the
low-frequency ratio; the 0.20 threshold sits below the fifth percentile of the
reference clips and above the field false-positive maximum, retaining the large
majority of genuine calls while still removing the insect false positives.
```

**Why it is false:** the history is stated backwards — `src/config.py:249` ships 0.40 as the **current** value with the comment "(was 0.20)", and `src/config.py:224-225` tabulates "0.20 ← original, 0.40 ← current". 0.40 was never "too aggressive"; it costs 4.2 points of reference-clip recall. And "the 0.20 threshold … [sits] above the field false-positive maximum" is flatly contradicted by the deployment: the field false-positive maximum is 0.98 and the minimum is 0.2007. As written, the paper argues against the value its own code uses.

**Replacement (lines 1110–1117):**
```latex
low-frequency energy gate at detection time (see \S\,Sliding-window detection)
intended to catch high-frequency false positives whose energy lies almost
entirely above 1.5\,kHz. The deployment reported here used a threshold of 0.20,
chosen because the \textit{Colobus\_confuser} clips it was calibrated on have
low-frequency ratios of at most 0.092. The field false positives proved not to
be of that kind: all 253 \textit{Colobus} detections have ratios above 0.20,
with a median near 0.40 and a maximum of 0.98, so the gate as deployed removed
none of them. The threshold has since been raised to 0.40, which removes
roughly half of the 253 at a cost of 4.2 points of reference-clip recall
(97.6\,\% to 93.4\,\%). Because no field detection has ever been confirmed as a
genuine roar, the recall side of that trade is unmeasured, and the gate is a
mitigation rather than a fix.
```

---

## B3 — The entire validation-set block is a measurement of leakage (lines 632–718)

`src/train.py:186-194` documents the defect and names it: the split is taken after 7× augmentation (9× for Colobus), so P(all variants land on one side) = 0.8⁷ = 0.21, i.e. ~79 % of clips leak. For Colobus it compounds: 617 windows cut at a 1 s hop from **172** source recordings. Every cell of `tab:val` and `tab:confusion` is arithmetically self-consistent and consistently meaningless (518+1111+915+1173 = 3717 is the augmented-pool size; 3647/3717 = 98.12 %).

There is **no honest recomputation** of a curated-clip accuracy. Do not substitute `grouped_val_accuracy`: it is per-fold, its validation side is 86.1 % Background (majority baseline 0.861), and it is not comparable across folds.

### B3a — Framing sentence (lines 632–635)

**Current:**
```latex
We report validation at two levels: (i) performance on a held-out validation
split drawn from the same curated clip pool as training, and (ii) deployment on
the continuous field recordings, where the biology of the site provides an
independent check on the \textit{Colobus} false-positive rate.
```
**Why wrong:** level (i) is not held out; level (ii) promises a Colobus check the paper never delivers.

**Replacement:**
```latex
We report validation at one level: the continuous field recordings. The
validation split used during development is not held out --- it was drawn after
augmentation, from the same source recordings --- and \S\,Why the curated-clip
accuracy is not reported explains why its accuracy is not a generalization
estimate. The field measurement is a 16-fold leave-one-station-out sweep
against a manual review of every exported detection. The \textit{Colobus} class
is treated separately: because no \textit{C.\ guereza} vocalisation has been
confirmed at this site, every \textit{Colobus} detection is by construction a
false positive, which yields a false-positive rate but no true positives and
therefore no \textit{Colobus} detection-performance figure.
```

### B3b — Delete lines 637–647 and replace

**Current (lines 637–647):**
```latex
\paragraph{Validation-set performance (V12 model).}
The four-class V12 model (the frequency-position-aware head with
high-frequency nuisance augmentation) was evaluated on a stratified
validation split. The model was trained in two stages
(Fig.~\ref{fig:training}): a frozen-base stage that trains only the
classifier head, followed by a fine-tuning stage in which the last two VGG19
blocks are unfrozen. Both stages use early stopping and learning-rate
reduction on a validation-accuracy plateau; the deployed checkpoint is the
epoch with the highest validation accuracy (98.12\,\%).
Table~\ref{tab:val} summarizes per-class results and
Table~\ref{tab:confusion} gives the full confusion matrix.
```

**Replacement:**
```latex
\paragraph{Why the curated-clip accuracy is not reported.}
The V12 model was originally scored on a random 80/20 split of the augmented
clip pool, which returned 98.12\,\% accuracy. That figure is not a
generalization estimate and we do not report it as one. The split was taken
after augmentation, so approximately 79\,\% of source clips placed a
near-duplicate of a validation image into training ($0.8^{7}=0.21$), and the
\textit{Colobus} class compounds this because its 617 windows are cut at a
1\,s hop from 172 source recordings. The gap between that number and the
0.4096 precision the same model achieved on 6\,189 reviewed field detections is
measurement error, not a domain shift to be explained away. Under the grouped
split now used (\texttt{src/train.py}, \texttt{grouped\_split}), a per-fold
validation accuracy is in any case uninformative: the validation side is
86.1\,\% Background, so a majority-class baseline scores 0.861, and the split
differs between folds. We therefore replace curated-clip accuracy with the
station-level field measurement reported below.
```

### B3c — Figure caption (lines 655–656)

**Current:**
```latex
two VGG19 blocks (Stage~2). The deployed model is the best-validation-accuracy
checkpoint (98.12\,\%).}
```
**Replacement:**
```latex
two VGG19 blocks (Stage~2). This figure is included to show optimization
behaviour only --- convergence, the effect of the learning-rate drop, and the
absence of divergence at the stage transition. The validation curve was
computed on a split taken after augmentation and is inflated by near-duplicate
leakage; its height carries no information about generalization and should not
be read as a performance figure.}
```

### B3d — `DELETE` lines 661–678 (`tab:val`) and lines 680–696 (`tab:confusion`)

Replace `tab:val` with the leave-one-station-out table in §3 of this document. Delete `tab:confusion` outright — under a grouped split no honest confusion matrix exists, and `colobus_in_eval == 0` in all 16 folds, so no cross-species cell can be populated from field data.

### B3e — `DELETE` lines 698–706 (`fig:confusion`, `confusion_matrix_v12.pdf`)

The headline claim on the caption, "Cross-species confusion between Cernic and *Colobus guereza* is exactly zero", separates 397 site-recorded Cernic clips from 1111 images derived from 172 Macaulay Library recordings. That is a microphone-and-background separation, not a species separation, and it would very likely hold for two arbitrary sounds drawn from those two archives.

Nothing else in the .tex references `tab:confusion` or `fig:confusion`; the two deletions leave no dangling `\ref`.

### B3f — Replace the prose at lines 708–718

**Current:**
```latex
\noindent Overall accuracy is 98.12\,\% (loss 0.0994) across 3717 validation
clips (Fig.~\ref{fig:confusion}). Two features of the confusion matrix are
central to the method. First,
cross-species confusion between the two primate classes is exactly zero: no
Cernic clips are called \textit{Colobus} and no \textit{Colobus} clips are
called Cernic (0 of 1629 primate clips). Second, the dedicated confuser class
is well separated from real \textit{Colobus}: of 1111 genuine \textit{Colobus}
clips only 5 are assigned to the confuser, and of 915 confuser clips only 2
are assigned to real \textit{Colobus}. Because the confuser group is folded
into Background at detection time, these confuser windows do not produce
\textit{Colobus} detections.
```

**Replacement:** the prose given in §3 below ("The macro-average over the 16 station folds …").

---

## B4 — "Manual verification is still in progress" is false (lines 740–744)

**Current (verbatim, lines 740–744):**
```latex
The final (V12) model was deployed across 16 acoustic stations. This section
reports the \textit{Cercopithecus nictitans} detections, every one of which was
verified by ear; \textit{Colobus guereza} detections were produced by the same
run but their manual verification is still in progress and is not included
here. The model produced 6\,189 \textit{C.\ nictitans} detections across the 16
```

**Why it is false:** the verification is complete. All 253 were listened to; none is a genuine roar (`src/config.py:215-216`). This is the single most damaging sentence in the paper: it asserts that a finished, wholly negative review has not happened. `paper/SUBMISSION_CHECKLIST.md:52-54` carries the same stale claim.

**Replacement:**
```latex
The final (V12) model was deployed across 16 acoustic stations. This section
reports the \textit{Cercopithecus nictitans} detections, every one of which was
verified by ear. The same run produced 253 \textit{Colobus guereza} detections;
these have also all been verified by ear, and not one is a genuine roar. They
are reported below as a negative result, and no \textit{C.\ guereza} precision,
recall, or activity figure is given, because the confirmed count is zero. The
model produced 6\,189 \textit{C.\ nictitans} detections across the 16
```

---

## B5 — "Residual Colobus false positives" describes a 100 % failure rate as a remainder (lines 1126–1130)

**Current (verbatim, from mid-line 1126 to the start of 1130):**
```latex
a genuine call. This is the root cause of the residual \textit{Colobus} false
positives, and it is why a high confidence threshold alone does not remove them;
the dedicated confuser class and the frequency-position encoding target this
failure mode but cannot be expected to eliminate every out-of-distribution
sound.
```

**Why it is false:** there is no residue. 253 of 253 Colobus detections are false — the entire Colobus output — at median confidence 0.953. `src/config.py:236-238` states the honest reading: a detector that fires confidently on thunder has a training problem. The word also appears at line 1111 and is removed by B2b.

**Replacement:**
```latex
a genuine call. This is the root cause of the \textit{Colobus} false positives,
which are not a residue but the entire \textit{Colobus} output: all 253
\textit{Colobus} detections returned by the 16-station deployment were reviewed
by ear and none was a genuine roar, at a median detection confidence of 0.95.
That confidence is why a high threshold alone does not remove them. The
dedicated confuser class and the frequency-position encoding target this
failure mode but did not eliminate it, because the out-of-distribution sound
the field supplied---thunder---is low-frequency, and so resembles the target in
the one dimension all three defences measure.
```

---

## B6 — The hard-negative loop is presented as a contribution; it is what corrupted the labels (lines 578–586, and README.md:111-115)

**Current (verbatim, lines 578–586):**
```latex
\subsection*{Iterative hard-negative mining}

The exported strong false positives are the sounds the detector most needs to
learn to ignore. They are added to the Background class and the model is
retrained from the preprocessing stage, so that on the next pass the network
is far less likely to fire on those sounds. Repeating the detect $\rightarrow$
clean $\rightarrow$ fold-in $\rightarrow$ retrain loop a few times (typically
three to five iterations) progressively drives down the false-positive rate
without ever requiring exhaustive manual annotation of the field recordings.
```

**Why it is false:** the claimed benefit is unmeasured, and the measured effect is the opposite. `scripts/build_v13_dataset.py:8-12`: of the 368 auto-flagged clips that appear in the human review, **68 are confirmed genuine calls (18.5 %)**, some at confidence 0.99–1.00, and 20 of them come from IPA19/IPA20 — the two stations the configuration declares held out. A blind second listener re-judged those 68 and called 64 of them genuine (`data/labels/disputed_68_labels.csv`, 94.1 % agreement with the reviewer); all 44 clips contributed by the Mahalanobis filter were among the genuine ones. `src/config.py:81` lists `outputs/auto_cleanup/auto_flagged_fp` inside `BACKGROUND_FOLDERS` and `src/data_loader.py:37` uses `os.walk`, so a fresh preprocessing run reloads that folder as Background with **no action from the user**.

**Replacement:**
```latex
\subsection*{Hard-negative mining, and why it must not be automatic}

The exported false positives are, in principle, the sounds the detector most
needs to learn to ignore, and the pipeline implements a detect $\rightarrow$
clean $\rightarrow$ fold-in $\rightarrow$ retrain loop that adds them to the
Background class and retrains from the preprocessing stage. We ran that loop,
and we report it here as a negative result: driven by the automatic flags, it
silently poisons the training set. The flags come from the same model whose
errors they are meant to correct, and the filters that produce them are weak
(\S\,Method validation), so the export is not a clean sample of false
positives. Cross-matching the flagged pool against the human review shows that
68 of the 368 flagged clips that appear in the review are confirmed genuine
calls (18.5\,\%), some at confidence 0.99; a second listener, blind to the
original verdicts, judged 64 of those 68 genuine. Folding such an export into
Background teaches the detector to ignore the calls it exists to find, and
because the export folder is listed among the background folders by default, a
fresh build reloads it without the user doing anything.

The loop is therefore kept in the code but must be run with a human in it: the
exported clips are a listening queue, not labels. We have no measurement
showing that the automatic version reduced the field false-positive rate, and
the labels it produced had to be rebuilt from human listening before the model
could be evaluated honestly.
```

**Companion edit, contributions list (lines 168–177).** Replace `(iii) an iterative hard-negative mining loop that recycles confirmed false positives` — the loop recycles *automatically flagged* clips, and "signals independent of the detector itself" is false for three of the four signals (confidence, softmax margin and Mahalanobis distance all come from the detector; only the neighbour count is independent):
```latex
Rather
than only training a classifier, we wrap it in (i) a detection stage with
domain-appropriate post-processing, (ii) an automatic cleanup that ranks and
groups detections for review using signals the detector already produces
together with one, the temporal neighbour count, that is independent of it, and
(iii) a hard-negative mining loop that recycles \emph{human-confirmed} false
positives into the background class and retrains --- reported here with a
caution, because running the same loop on automatic flags corrupted our own
training labels.
```

**Companion edit, Limitations (lines 1136–1141).** "may be systematically hidden … this is mitigated by a mining-confidence audit" understates a measured, unmitigated failure:
```latex
The automatic cleanup carries its own caveats, and the most serious of them is
not hypothetical. It introduces a circular bias, because the hard negatives are
scored by the same model that produced them, so genuine calls the model scores
low are routed into the class that teaches it to score them lower still. We
measured this rather than assuming it away: a blind relabelling of 68 clips
drawn from the automatic false-positive export found 64 of them to be genuine
calls, and all 44 of the clips contributed by the Mahalanobis filter were among
them. The audit did not mitigate the bias; it established that the automatic
export cannot be used as training labels without listening to it first.
```

---

## B7 — The Colobus reference set is unattributed third-party media (line 237, plus bibliography)

**Current (verbatim, line 237):**
```latex
Colobus\_guereza  & 617          & \textit{Colobus guereza} reference calls \\
```

**Why it blocks submission:** the string "Macaulay" appears **nowhere** in `methodsx_manuscript.tex`, `README.md` or `SETUP.md`. `scripts/fetch_colobus_library.py:1-2` fetches these recordings from the Macaulay Library, and lines 28-29 of that script state that Macaulay media must be cited by recordist and asset ID in any publication. The 617 windows collapse to exactly 172 unique source prefixes. This is a licence condition, not a citation-style preference.

**Replacement (line 237, plus a table note and a `\bibitem`):**
```latex
Colobus\_guereza  & 617          & \textit{Colobus guereza} reference windows,
                                   cut from 172 Macaulay Library archival
                                   recordings \cite{macaulay}; none from the
                                   study site \\
```
Immediately after `\end{tabular}` in that table, before `\end{table}`:
```latex
\vspace{2pt}
\footnotesize\noindent\textit{Note.} \textit{Colobus guereza} reference audio
was obtained from the Macaulay Library, Cornell Lab of Ornithology. Asset IDs
and recordists for all 172 source recordings are listed in Supplementary
Table~S1. We thank the contributing recordists.
```
New bibliography entry:
```latex
\bibitem{macaulay}
Macaulay Library, Cornell Lab of Ornithology, Ithaca, NY.
\url{https://macaulaylibrary.org}. Individual asset IDs and recordists are
given in Supplementary Table~S1.
```

**Companion edit, class list (line 209):**
```latex
\item \textbf{Colobus\_guereza}: \textit{Colobus guereza} reference calls,
617 windows cut at a 1\,s hop from 172 Macaulay Library archival recordings
made away from the study site. No \textit{C.\ guereza} vocalisation has been
confirmed at Ipassa-Makokou, so this class has no site-recorded positive
example and the domain shift between reference and field channel is
unquantified;
```

**Companion edit, Introduction (lines 159–163):** "two primate species recorded at Makokou" is false for *C. guereza*. Append after "\textit{Colobus guereza}.":
```latex
Only \textit{C.\ nictitans} is recorded at the site: the \textit{C.\ guereza}
reference set is drawn entirely from archival recordings made elsewhere,
because no \textit{C.\ guereza} vocalisation has been confirmed at
Ipassa-Makokou.
```

---

## B8 — The V12 field table is partly in-sample at six of its sixteen stations (lines 220–224, affects Table `tab:field`)

**Current (verbatim, lines 220–224):**
```latex
\item \textbf{Background}: a pooled negative class assembled from ambient
forest noise, calls of non-target animals present at the site (e.g.\
\textit{Cercocebus torquatus}, \textit{Pan troglodytes}), and previously
misclassified clips recycled through the hard-negative mining loop.
```

**Why it blocks submission:** `BACKGROUND_FOLDERS` includes `outputs/auto_cleanup/auto_flagged_fp`; `scan_audio_files` recurses. That tree holds 4,344 wav files, 3,263 of them in station-named subfolders — `ipa2st_*` 1,377, `ipa14st_*` 582, `ipa13st_*` 420, `ipa11st_*` 391, `ipa16st_*` 266, `ipa10st_*` 227. IPA2ST, IPA10ST, IPA11ST, IPA13ST, IPA14ST and IPA16ST are six of the sixteen rows of `tab:field`, and their reported precisions are therefore partly in-sample. The leave-one-station-out language at lines 833–838 governs only the cleanup **cutoffs**, not the detector.

**I verified that this does *not* contaminate the V13 LOSO result.** `scripts/build_v13_dataset.py` attributes a provable station (or an explicit multi-station candidate list) to every clip, and `train_v13_loso.py:154-159` withholds a clip whenever the held-out station appears in its `possible_stations`. The build script's own header (`:22-25`, `:90-101`) explains that without this "every reported gain is in-sample". So the fix is: keep the V13 numbers, disclose the V12 contamination.

**Replacement:**
```latex
\item \textbf{Background}: a pooled negative class assembled from ambient
forest noise, calls of non-target animals present at the site (e.g.\
\textit{Cercocebus torquatus}, \textit{Pan troglodytes}), and previously
misclassified clips recycled through the hard-negative mining loop. The
recycled clips are field false positives mined from six of the sixteen stations
reported in \S\,Method validation (IPA2ST, IPA10ST, IPA11ST, IPA13ST, IPA14ST
and IPA16ST; 3\,263 clips), so the per-station V12 field precisions in
Table~\ref{tab:field} are partly in-sample at those six stations. The
leave-one-station-out evaluation reported below is not affected: it rebuilds
the training set with a provable station attached to every clip and withholds
every clip that could have come from the held-out station.
```

---

## B9 — No section states which time-of-day regime the field figures were computed under (insert at line 740)

Zero hits in the .tex for `TIME_FILTER`, `05:00`, `19:00`, `05:30`, `10:30`, "time of day". Meanwhile `src/config.py:321-322` ships `TIME_FILTER_START = "05:00"` / `TIME_FILTER_END = "19:00"`. Every field figure in the paper — 6,189; 2,535; 41.0 %; all of `tab:field` — is the **gate-off** row. A reader running the released code reproduces none of it. Insert after the scope sentence corrected in B4:

```latex
\textbf{All field figures reported in this section were computed with the
file-level time-of-day filter disabled} (\texttt{TIME\_FILTER\_START} and
\texttt{TIME\_FILTER\_END} set to \texttt{None}), so that every recording was
processed. The released configuration ships that filter enabled with a
05:00--19:00 window, calibrated as the widest window retaining $\geq$95\,\% of
confirmed calls; enabling it on the same audio yields 3\,571 detections, 2\,503
confirmed calls and a precision of 70.1\,\%, removing 70.8\,\% of the false
positives at a cost of 1.3\,\% of the calls. We report the unfiltered figures
here because they are the harder test of the detector and the correct ground
truth against which the cleanup is scored; the leave-one-station-out results
below are reported inside the 05:00--19:00 window, as the deployed
configuration would produce them.
```

Add this guard comment beside the time-filter text in §Sliding-window detection:
```latex
%% GUARD: the 05:00-19:00 window was FITTED to the reviewed detections. Any
%% statement in this paper about WHEN C. nictitans calls must be computed with
%% the filter OFF, or it merely recovers the window we imposed. 32 of the 2,535
%% confirmed calls fall outside 05:00-19:00. Do not add a diel result without
%% re-running gate-off.
```

---

## B10 — Forward-selection numbers are stale in all four values (lines 930–938)

**Current (verbatim):**
```latex
Added to the four signals already in use, however, these features gain nothing
that survives scrutiny. Forward selection run leave-one-station-out improves
mean average precision by 0.011 (bootstrap 95\,\% CI $-0.004$ to $+0.027$), and
two of the fifteen stations supply almost all of it: setting those two aside
leaves 0.002. Twelve of fifteen stations improved, which is not evidence when
the improvements are this small.
```

**Why wrong:** re-running the shipped `scripts/rank_signals_experiment.py --matched data/outputs/auto_cleanup/cleanup_vs_review.csv --exclude-station IPA4ST` gives mean +0.0152, median +0.0074, t = 1.58, bootstrap 95 % CI [−0.0012, +0.0353], 13/15 improved, +0.0045 with the two largest contributors set aside. A reader running the repo's own script gets four different numbers from the paper.

**Replacement:**
```latex
Added to the four signals already in use, however, these features gain nothing
that survives scrutiny. Forward selection run leave-one-station-out improves
mean average precision by 0.0152 (median $+0.0074$, $t=1.58$, bootstrap
95\,\% CI $-0.0012$ to $+0.0353$), and two of the fifteen stations supply most
of it: setting those two aside leaves $+0.0045$. Thirteen of fifteen stations
improved, which is not evidence when the interval still spans zero.
```

**Companion edit, lines 914–923 — the paragraph omits the two best candidates.** `src/episode_features.py` defines nine signals (`:41-52` six episode-scale, `:62-66` three event-scale), not six. The best single timestamp-only feature, `event_windows`, reaches AP 0.9039 under random tie-breaking (0.9098 stable-sorted) — above the whole four-signal ordering at 0.8997. Omitting it in a paragraph arguing these features add nothing is the omission a referee will find, because it is one script invocation away.
```latex
We used that to ask whether the temporal
structure the episodes expose adds anything as a per-detection signal, deriving
nine further quantities from the detection times alone: at the scale of a bout,
the size, duration, internal spacing and density of the episode a detection
belongs to, its position within that episode, and the distance to the nearest
same-species detection; and at the scale of a single vocalization, the number
of consecutive windows the detection's call event spans, that event's duration,
and the detection's position within it. Two are strong on their own. The number
of windows an event spans reaches an average precision of 0.904 across the
fifteen stations, above the 0.900 of the four-signal ordering itself; the size
of the episode a detection sits in reaches 0.827, not far below the 0.845 of
the detector's own confidence. (The event count is an integer that 24\,\% of
detections share, so it must be scored with random tie-breaking: a stable sort
inflates it to 0.910.) In both cases,
```

---

## B11 — README carries the withdrawn 98.12 % as its headline (README.md:8, :257-263, :24, :100-103, :111-115, :254-255)

The repository is what a reviewer opens. Six edits, all outside the .tex:

- **:8** — replace `(temporal_freqpos, 98.12% validation accuracy)` with the LOSO sentence and an explicit withdrawal.
- **:257-263** — replace the five-row per-class accuracy table with the three-row LOSO summary (macro / IPA18ST / IPA7ST) and the "precision at matched 95 % recall, not recall" disclaimer.
- **:24** — `~80 MB` → `~142 MB` (actual size 148,829,048 bytes).
- **:27-33 and SETUP.md:120-122** — the two `[link to be added]` placeholders point at each other; a reader cannot obtain the model at all. Either publish to Zenodo or state plainly that weights are by email request.
- **:100-103** — "Three filters … no manual listening needed": only two run (`USE_YAMNET_FILTER = False`), and cleanup moved precision 41.0 → 45.0 %, so listening is still required. The useful output is the queue (6,189 detections → 1,529 listens).
- **:111-115** — Step 5 still instructs the reader to run the loop that corrupted the labels, with no listening step, and `config.py:81` means they do not even have to follow it. Add the warning block and the "listen first" three-step procedure.
- **:254-255** — gate threshold `0.20` → `0.40`.

---

# 2. WRONG NUMBERS

| Location | Stated | Correct | Source |
|---|---|---|---|
| tex 74, 645, 656, 675, 708; README 8, 257-263 | 98.12 % validation accuracy | withdrawn — no honest replacement exists | `src/train.py:186-194` |
| tex 708 | loss 0.0994, n = 3717 clips | 3717 is the augmented-pool size, not a clip count | `tab:val` arithmetic, 518+1111+915+1173 |
| tex 711 | "0 of 1629 primate clips" | 1629 = 518 + 1111, augmented; and it is a field-vs-archive domain separation | `data/species/` layout |
| tex 670-675 | 96.14 / 99.01 / 97.81 / 98.38 | all leaked-split; delete | as above |
| tex 459, 1110; README 255 | gate threshold 0.20 | **0.40** | `src/config.py:249` |
| tex 1113 | 0.40 was "an earlier version … too aggressive" | 0.40 is the current committed value; 0.20 is the abandoned one | `src/config.py:224-225, :249` |
| tex 463 | field FP max low-freq ratio ≈ 0.09 | 0.98 (min 0.2007, median ≈0.42, p75 ≈0.84) | 253 Colobus rows, `data/outputs/detections/**/*.csv` |
| tex 465 | gate "removes essentially all of these false positives" | removed **0 of 253** | same |
| tex 1111, 1126 | "residual" Colobus false positives | 253 of 253 = 100 %, median confidence 0.953 | same |
| tex 742 | Colobus verification "still in progress" | complete; zero genuine roars | `src/config.py:215-216` |
| tex 931 | forward selection +0.011 | **+0.0152** | `scripts/rank_signals_experiment.py` |
| tex 932 | CI −0.004 to +0.027 | **−0.0012 to +0.0353** | same |
| tex 934 | "leaves 0.002" | **+0.0045** | same |
| tex 934 | "Twelve of fifteen stations" | **thirteen of fifteen** | same |
| tex 916 | "six further quantities" | **nine** (6 episode + 3 event) | `src/episode_features.py:41-52, 62-66` |
| tex 891, 846 | AP 0.900 | 0.8997 stable-sorted; must declare tie-break rule (24 % share one integer) | `scripts/rank_signals_experiment.py` |
| tex 1161 | "Cernic recall is 96.1 %" | no recall figure exists; 0.953 is *calls retained relative to V12* | `data/outputs/v13_loso.csv` |
| tex 239 | Background ≈5,900 | **6,295** raw (caption says "raw clip counts") | `BACKGROUND_FOLDERS` recursive count |
| tex 241 | Total ≈7,541 | **7,936** raw | same |
| tex 194-198 | "IPA1ST, IPA2ST, …" (implies 20) | **16** stations; 3, 5, 9, 12 were never deployed | `cleanup_vs_review.csv` |
| tex — absent | (corpus extent never stated) | **four consecutive days, 22–25 Feb 2021** | same |
| tex 122-124 | YAMNet listed as a constituent method | implemented but **disabled**; its three output columns are constants | `src/config.py:264` |
| README 24 | model ≈80 MB | **142 MiB** (148,829,048 bytes) | `data/outputs/models/best_model_v12.h5` |

**Two numbers that are correct — do not "fix" them.** (a) Cernic 370 at line 235 matches `SPECIES_FOLDERS` exactly (25+101+124+101+19); `HANDOFF.md:134`'s 397 wrongly adds a 5 s folder V12 never loads. (b) Every cell of `tab:ablation` reproduces exactly from `cleanup_vs_review.csv`; only its framing needs the caveat below.

**One correction to propagate into your own notes.** "Mahalanobis AUC 0.5154 = chance" is true of the **thresholded flag** only — 0.5154 = (229/3654 + 1 − 81/2535)/2. The continuous `mahalanobis_d2` is **not** chance: pooled AUC 0.7627, macro-average 0.7052, AP 0.782 as a ranking signal. If the paper says flatly that Mahalanobis is chance, a referee who computes the column's AUC gets 0.76 and the correction itself becomes the error. The defensible sentence is: *the thresholded filter is near-chance; the distance underneath it is a real but weak ranking signal, which is why we keep it in the ordering and drop it as a filter.* Add to the `tab:ablation` caption:

```latex
The two filters retain the same 95.3\,\% of genuine calls in either
case; what changes is how many false positives they reach. Nearly all of the
effect is temporal isolation: the Mahalanobis rows move precision by 0.7 and
0.3 points, its flag has an ROC area of 0.515 over all sixteen stations, and a
blind re-listen of 44 clips it had exported as false positives judged all 44
genuine calls. Those rows should be read as the negative result they are.
```

---

# 3. THE ONE ADDITION THE PAPER NEEDS

The V13 leave-one-station-out sweep is the honest replacement for everything B3 deletes. I re-derived every cell below from `data/outputs/v13_loso.csv` this session: 16 rows, `detections` sums to exactly 6,189, `gated_detections` to 3,571, macro means 0.6953 / 0.9340 / 0.8208 / 0.9530.

Insert immediately after the B3b paragraph.

```latex
\paragraph{Leave-one-station-out field validation.}
We re-trained the classifier 16 times, once per acoustic station, each time
withholding every training clip that could have come from that station and
scoring only that station's reviewed detections
(\texttt{scripts/train\_v13\_loso.py}; \texttt{data/outputs/v13\_loso.csv}).
Attribution is not optional here: a clip whose station cannot be proved is
excluded from every fold it might belong to, because a single mis-attributed
clip would let a fold quietly include the station it claims to hold out. The
held-out folds partition the 6\,189 reviewed detections exactly. Scoring is
restricted to the recalibrated deployment window 05:00--19:00, which retains
3\,571 of the 6\,189 detections and 2\,503 of the 2\,535 confirmed calls.
Table~\ref{tab:loso} reports, per station, the reviewed precision of the
deployed V12 detections inside that window, and the precision after rescoring
with the held-out model at the per-fold threshold chosen to retain 95\,\% of
the calls V12 already found.

Four limits must be read before the numbers.
First, this is a \emph{precision} measurement at matched recall on the windows
V12 fired on. It is not a recall measurement, it says nothing about calls
neither model produced, and it is not a substitute for a validation accuracy.
Second, the review that supplies the labels covers four consecutive days
(22--25 February 2021) and is 100\,\% \textit{C.\ nictitans}; nothing here
generalises to other seasons.
Third, the \textit{Colobus} class contributes no evaluation positive in any of
the 16 folds, because the field record contains none, so this table measures
nothing about that species.
Fourth, stations are averaged unweighted, so a station contributing 82
detections counts as much as one contributing 520; the macro-average is
therefore an estimate for a \emph{typical} station, not for the pooled corpus.
```

```latex
\begin{table}[H]
\centering
\caption{Leave-one-station-out validation, one fold per acoustic station.
All figures are computed inside the deployed time window (05:00--19:00) on the
detections the V12 deployment exported and a human reviewed;
$n_{\mathrm{gated}}$ is the number of such detections at that station (3\,571
of the 6\,189 reviewed in total). ``V12 prec.'' is the reviewed precision of
those detections. ``V13 prec.'' is the precision after rescoring with the
held-out model at the per-fold threshold that retains 95\,\% of the calls V12
found; ``FPs removed'' and ``Calls kept'' are the realised fractions at that
threshold. These are precision figures at matched recall on windows V12 fired
on --- not recall, and not a validation-accuracy substitute. The corpus is four
consecutive days and 100\,\% \textit{C.\ nictitans}; no fold contains a
\textit{Colobus} positive.}
\label{tab:loso}
\begin{tabular}{lrrrrr}
\toprule
Station & $n_{\mathrm{gated}}$ & V12 prec. & V13 prec. & FPs removed & Calls kept \\
\midrule
IPA1ST  & 277 & 0.437 & 0.943 & 0.955 & 0.950 \\
IPA2ST  &  82 & 0.781 & 0.968 & 0.889 & 0.953 \\
IPA4ST  & 131 & 0.695 & 0.946 & 0.875 & 0.956 \\
IPA6ST  & 171 & 0.883 & 0.966 & 0.750 & 0.954 \\
IPA7ST  & 114 & 0.605 & 0.759 & 0.533 & 0.957 \\
IPA8ST  & 257 & 0.599 & 0.987 & 0.981 & 0.955 \\
IPA10ST & 116 & 0.888 & 0.970 & 0.769 & 0.952 \\
IPA11ST &  93 & 0.452 & 0.909 & 0.922 & 0.952 \\
IPA13ST & 238 & 0.773 & 0.941 & 0.796 & 0.951 \\
IPA14ST & 236 & 0.661 & 0.949 & 0.900 & 0.955 \\
IPA15ST & 189 & 0.868 & 0.957 & 0.720 & 0.951 \\
IPA16ST & 324 & 0.565 & 0.874 & 0.823 & 0.951 \\
IPA17ST & 520 & 0.696 & 0.905 & 0.772 & 0.950 \\
IPA18ST & 182 & 0.681 & 0.992 & 0.983 & 0.952 \\
IPA19ST & 196 & 0.607 & 0.905 & 0.844 & 0.958 \\
IPA20ST & 445 & 0.935 & 0.973 & 0.621 & 0.952 \\
\midrule
\textbf{Macro mean} & --- & \textbf{0.695} & \textbf{0.934} & \textbf{0.821} & \textbf{0.953} \\
\bottomrule
\end{tabular}
\end{table}
```

```latex
\noindent The macro-average over the 16 station folds is a starting precision
of 0.695 and a rescored precision of 0.934, obtained by removing 82.1\,\% of
the reviewed false positives while retaining 95.3\,\% of the reviewed calls
(Table~\ref{tab:loso}). The spread across stations matters more than the mean.
The strongest station, IPA18ST, moves from 0.681 to 0.992 with 98.3\,\% of its
false positives removed; the weakest, IPA7ST, moves from 0.605 to only 0.759
with 53.3\,\% removed, so at that station rescoring still leaves one false
positive in four. A per-station reading of the table, not the macro-average, is
what a user should plan against. Note also that IPA4ST, which alone produced
2\,470 of the 6\,189 unfiltered detections, contributes only 131 detections
inside the deployment window: its insect chorus is nocturnal and the time gate,
not the classifier, is what removes it.

The gain is not architectural. The classifier head is identical between the
deployed model and the retrained one --- the same frequency-position-aware
head, 1\,972\,740 head parameters, in both. The deployed model additionally
fine-tuned the last two VGG19 blocks (12\,297\,732 trainable parameters); the
leave-one-station-out models hold the trunk at stock ImageNet weights and train
the head alone. The improvement is therefore obtained with strictly less
trainable capacity, and is attributable to the training labels and the
evaluation protocol rather than to the model.

The \textit{Colobus} class is not measured by this table, and cannot be. No
fold contains a confirmed \textit{Colobus} call, because the field record
contains none: all 253 \textit{Colobus} detections exported by the deployment
were listened to and none was a genuine roar. All \textit{Colobus} training
positives come from Macaulay Library archival recordings; none is from the
site. We therefore report the \textit{Colobus} branch as a false-positive
control only --- the low-frequency gate, recalibrated to 0.40, removes roughly
half of the 253 --- and make no claim about its detection performance.
```

**Companion edit at line 415** (fine-tuning protocol), so V13's gain is not attributed to fine-tuning it does not do:
```latex
unfrozen, while block1 and block2 remain frozen to preserve
low-level features; this makes $\approx$12.3\,M parameters trainable. The
leave-one-station-out models reported under Method validation omit this stage
entirely: they hold the VGG19 trunk at stock ImageNet weights and train only
the 1\,972\,740-parameter head. The
```

**Companion edit at lines 419–422** (the leak, currently stated as protocol — and no longer describing the released code, which calls `grouped_split`):
```latex
\paragraph{Shared training settings.}
Both stages use an 80/20 train/validation split \emph{grouped by source
recording}, with a fixed random seed for reproducibility. Grouping is not
cosmetic. Each source clip yields seven augmented images (nine for
\textit{Colobus}), and the \textit{Colobus} windows are cut from their source
recordings at a 1\,s hop, so a split taken over the augmented pool leaves
near-duplicates of validation images in training: the probability that all
seven variants of a clip fall on the same side is $0.8^{7}=0.21$, i.e.
approximately 79\,\% of clips leak. Grouping places every image derived from
one recording on the same side of the split
(\texttt{src/train.py}, \texttt{grouped\_split}).
```

**Companion edit at lines 1160–1165** (the recall sentence, which currently quotes the leaked 96.1 % and then disclaims it):
```latex
Finally, the operating point favours precision over recall and is therefore
conservative on weak or short calls. No absolute recall figure is available:
the only recall quantity we can defend is relative, namely that the
leave-one-station-out threshold retains 95.3\,\% of the calls the deployment
already found (macro-average over 16 stations), so rescoring discards about one
call in twenty of those it inherits. That says nothing about calls neither
model produced. The fixed 2\,s analysis window would also truncate or split
vocalizations longer than it, such as \textit{Colobus} roars, were any to be
recorded at these stations.
```

**New Limitations paragraph** — the paper currently offers no explanation for a 100 % false-positive rate on one of its two named species. Insert after the existing closed-set paragraph:
```latex
\paragraph{The \textit{Colobus} branch is a negative result.}
The \textit{C.\ guereza} reference set contains no audio from the study site:
all 617 training windows are cut from 172 Macaulay Library archival recordings
made elsewhere, at different distances and through a different recording
channel. Two very different situations therefore produce the same zero ---
either \textit{C.\ guereza} does not vocalise at these stations, or it does and
the library-trained weights do not transfer to this channel --- and the
deployment cannot distinguish them. What the deployment does establish is that
every \textit{Colobus} detection it produced was spurious (253 of 253, median
confidence 0.95), and that a probe recording the raw \textit{Colobus} softmax
score on every window of dawn recordings, rather than thresholding it, finds
maximum scores of order $10^{-4}$: the model never approaches firing on a
genuine roar. We therefore report the \textit{Colobus} branch as a method that
has been built and deployed but not validated on a confirmed positive, and the
253 clips as hard negatives for the next model rather than as detections.
```

---

# 4. WHAT CANNOT BE FIXED BY EDITING

**4.1 The Macaulay asset IDs may not exist any more.** B7's LaTeX presumes a Supplementary Table S1 listing 172 recordist/asset-ID pairs. That list has to be recovered from whatever input was fed to `scripts/fetch_colobus_library.py`. If it is gone, the Colobus class cannot be published as described, and the only remaining options are to re-fetch with IDs recorded, or to remove the class from the paper. **Decide this before any other Colobus work** — it can make several of the edits above moot.

**4.2 The V12 field table needs recomputation or a retrain.** B8 discloses the contamination; it does not remove it. Two honest exits: (a) report field precision over the ten stations whose audio never entered training, as the headline, with all sixteen in an appendix; or (b) retrain V12 with the six stations excluded and regenerate `tab:field`. Option (a) is cheap and defensible. Note that `tab:field` also currently duplicates what `tab:loso` now says better; you may not need both.

**4.3 The Colobus scope decision is yours, not an accuracy fix.** After every edit above, each individual sentence is defensible, but a reviewer may still ask why a species with zero confirmed vocalisations appears in the keywords and the framing. The two coherent versions are: **(i)** two-species method whose second species is reported as an explicit negative result — what the edits above build; or **(ii)** single-species method with a Colobus confuser-control case study, dropping *C. guereza* from the abstract and keyword list. I did not write (ii) because it changes what the paper is about.

**4.4 Which time regime the paper reports.** B9 labels the figures gate-off and gives the gate-on alternative. But the LOSO table is gate-on and the cleanup/ablation tables are gate-off, so the paper currently reports two regimes side by side. Either accept that with the labels B9 adds, or regenerate the cleanup and ablation figures gate-on so the whole Method-validation section is one regime. That is a compute decision, not an edit.

**4.5 Every average-precision figure needs regenerating under a declared tie-break rule.** 24 % of detections share one integer value and `effort_curve` sorts stably, so the reported 0.900, 0.827 and 0.845 are all stable-sort artefacts to an unknown degree. Re-run `scripts/rank_signals_experiment.py` and `effort_curve` with an explicit random tie-break seed (≥200 draws), take whatever comes out, and state the rule in the `tab:ranking` caption. Do not hand-edit these numbers.

**4.6 There is no recall figure of any kind, and there will not be one without new work.** 0.953 is *calls retained relative to what V12 already found*. `scripts/recall_sample.py` exists; until it is run against a blind sample of un-detected audio, the paper must say plainly that absolute recall is unmeasured. Do not let 0.953 drift into being described as recall anywhere.

**4.7 Colobus recall is structurally unmeasurable at this site.** Not a wording problem. With no confirmed positive, no threshold, gate, or architecture change can be evaluated for its cost in true positives.

**4.8 Repository code changes.** `src/config.py:81` wires `outputs/auto_cleanup/auto_flagged_fp` into `BACKGROUND_FOLDERS`, so anyone reproducing V12 from the released code silently reloads 4,344 machine-labelled clips — including ~118 real calls and 20 clips from the declared held-out stations. Empty the folder or comment out that entry before the repository is cited by a published paper. The 61 WAVs currently in it are also a fossil: they were exported when the YAMNet cross-check was still enabled, and cannot be reproduced by the shipped two-filter configuration.

**4.9 The model weights are not obtainable.** Two mutually-referring `[link to be added]` placeholders (README.md:28, SETUP.md:120-122). Publish to Zenodo (which also gives you a DOI and lets you drop the "on reasonable request" hedge) or state the email-request route in both files.

**4.10 One claim that must never enter the paper.** "90.7 % of Colobus detections fall 19:00–05:00" is wrong — it was computed from 97 of 253 files because a regex matched only one of two coexisting filename schemes. The true value is 48.6 %, modal hour 15:00, 191 of 253 on a single day (2021-02-24), with two half-hours accounting for 54.2 % of all Colobus detections: a storm front. That figure appears nowhere in the .tex today (I confirmed the manuscript makes no diel claim at all) — keep it that way, and keep the guard comment from B9.

---

# 5. ORDER OF WORK

1. **Answer 4.1 first.** Whether the Macaulay asset IDs still exist determines whether the Colobus class can be published at all, which determines whether B1, B4, B5, B7 and the new Limitations paragraph are the right edits or whether you are writing version (ii) of 4.3. Everything Colobus-shaped waits on this.

2. **B4, then B1.** The "verification is still in progress" sentence is the one that converts an omission into a misrepresentation, and it is a five-line edit. Fix it before the abstract, because the abstract's replacement asserts the negative result that B4 establishes.

3. **B2a + B2b together, in one commit.** The paper currently argues against the value its own code ships and states a field maximum that its own CSVs refute by an order of magnitude. Landing only one half leaves an internal contradiction. Then fix README:255 in the same pass.

4. **B3 + §3 together, in one commit.** Deleting `tab:val`, `tab:confusion` and `fig:confusion` leaves a hole in Method validation; `tab:loso` fills it. Doing these separately produces an intermediate state with no validation section at all. This is the largest single edit — lines 632–718 plus three companion edits at 415, 419–422 and 1160–1165 — but it is also the one that turns the paper from indefensible to publishable.

5. **B8 and B9.** Both are disclosures, both are short, and both change how a reader interprets `tab:field`. Do them adjacent so the field-deployment section is internally consistent about what was and was not held out, and under what regime.

6. **B6 and B5.** The hard-negative-loop rewrite and the "residual" replacement. B6 retitles a section, so run a `\ref` check afterwards — line 1010's cross-reference to `\S\,Iterative hard-negative mining` dies with it and needs updating to `\S\,Hard-negative mining`.

7. **B10, then 4.5.** Fix the four stale forward-selection numbers and the six-versus-nine features immediately (they are already measured), then schedule the tie-break re-run for when the CPU job finishes and regenerate everything downstream of `effort_curve`.

8. **B11 and 4.8/4.9.** Repository last, but before the DOI is minted — the README is what a reviewer opens, and `config.py:81` is what a reproducer runs.

9. **Un-tick `paper/SUBMISSION_CHECKLIST.md:115.`** "Method details §1–§8 verified line-by-line against the code" is disproved by the gate mismatch and the absent time-filter regime. Re-tick only after step 8. Mark line 52-54 (`Colobus guereza field verification`) as **done, and negative**.

**The single highest-value hour** is step 4: it removes the only number in the paper a reviewer is certain to test, and replaces it with a stronger, defensible result that already exists on disk.