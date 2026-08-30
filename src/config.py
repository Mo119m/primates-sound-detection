"""
Configuration file for Primate Vocalization Detection Pipeline
All paths, parameters, and settings are defined here for easy modification.
When adding new species or updating data, only need to modify this file.
"""

import os

# DATA ROOT PATH
# PRIMATE_DATA_ROOT wins when set. Otherwise pick the layout that matches where
# we are running: the Google Drive folder in Colab, and the repository's own
# data/ folder anywhere else, so a local clone works with no configuration.
_COLAB_ROOT = "/content/drive/MyDrive/primates-data"
_REPO_DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _default_data_root() -> str:
    env = os.environ.get("PRIMATE_DATA_ROOT")
    if env:
        return env
    # /content/drive/MyDrive only exists once Drive is mounted in Colab.
    if os.path.isdir("/content/drive/MyDrive"):
        return _COLAB_ROOT
    return _REPO_DATA


DRIVE_ROOT = _default_data_root()
# In the current Drive layout the species/ and background/ folders live
# directly under DRIVE_ROOT, so AUDIO_ROOT == DRIVE_ROOT by default.
AUDIO_ROOT = os.environ.get("PRIMATE_AUDIO_ROOT", DRIVE_ROOT)
LONG_AUDIO_ROOT = os.environ.get(
    "PRIMATE_LONG_AUDIO_ROOT", os.path.join(DRIVE_ROOT, "long_audio")
)

# SPECIES CONFIGURATION
# Cercopithecus nictitans call types are merged into a single "Cernic" class
# for presence detection. Each entry can be a single folder (str) or a list of
# folders whose clips are pooled under one label.
SPECIES_FOLDERS = {
    'Cernic': [
        'species/CERNIC putty-nose 2s',
        'species/CERNIC hacks',
        'species/CERNIC keks',
        'species/CERNIC pyows',
        # Real Cernic calls that were wrongly mined into Background by the
        # auto-cleanup loop (the model fired high-confidence Cernic on a
        # dev-station window, it was assumed a false positive, but human review
        # confirmed a genuine putty-nose call). Recovered here as positives.
        # Label safety: every clip is human-verified AND confirmed to originate
        # from a dev station (IPA1-18 / Makokou short-term), never the held-out
        # IPA19/20 -- so no test-station audio leaks into training. Safe to list
        # before it exists; scan_audio_files() warns and skips a missing folder.
        'species/CERNIC field_confirmed',
    ],
    # Full-length windows carrying a run of roar pulses, not one pulse each.
    #
    # Three versions of this class have been tried and the difference is what a
    # training example is, not how many there are. Fixed 2 s crops (617) had the
    # right length but arbitrary phase: two crops of the same roar could be cut
    # half a pulse apart. Single pulses (665) fixed the phase and broke
    # something worse -- a putty-nosed hack is one 0.08 s event and a window
    # holding one is a faithful example, but a guereza roar is a train of
    # ~0.31 s pulses at ~1.7 Hz, so a window holding one pulse is a
    # low-frequency thump with 1.6 s of borrowed ambient around it. Trained on
    # those, the model could not learn the rhythm, and the symptom showed up in
    # deployment: it returned 0.9999 on clips whose energy was 99 % above
    # 1.5 kHz, which no roar can produce, because a lone thump in a bed was the
    # only thing it had ever been shown. Every one of its IPA4ST detections was
    # then discarded by the low-frequency gate -- the gate was carrying the
    # class.
    #
    # Bouts were tried and reverted, and the measurement that reverted them is
    # the useful part. Windows carrying three library pulses at ~1.7 Hz dropped
    # field sensitivity from 2/9 to 1/9 on the only field-verified guereza audio
    # in the project. Measuring those nine clips explains why: in the field a
    # roar arrives as ONE long low event, median one pulse per 3 s clip at
    # 0.7-2.5 s each, while the library material is a train of seven ~0.26 s
    # pulses at ~2 Hz. Distance smears the train into a rumble, and training on
    # the un-smeared train taught the model to require a rhythm the recorders
    # never deliver. "More complete call structure" is not the same as "closer
    # to the field", which is the lesson worth keeping.
    #
    # Single pulses are not right either -- they are what the low-frequency gate
    # ends up carrying -- but a lone 0.37 s event is closer to one long field
    # event than three fast ones are, and it is what the deployable model uses.
    # The real fix is to degrade the library material to the field's measured
    # structure rather than to re-cut it; the bout windows in
    # 'species/Colobus guereza bouts' are the right raw material for that,
    # because a smear needs a train to smear.
    'Colobus_guereza': 'species/Colobus guereza pulses',
    # Dedicated hard-negative class for the low-frequency forest sound that the
    # model repeatedly mis-fires as Colobus (pulsed, <1 kHz, morphologically
    # close to a guereza roar). These clips are ALL the Colobus false positives
    # mined by the auto-cleanup loop across dev stations -- human-reviewed so no
    # real Cernic leaks in (genuine Cernic found during review was deleted or
    # recovered into 'CERNIC field_confirmed'). Giving the confuser its own
    # softmax output forces the model to learn the Colobus-vs-confuser boundary
    # explicitly instead of drowning these few hundred hard negatives inside the
    # huge generic Background class (which V9 proved is not enough). At detection
    # time this class is folded into the Background group (see DETECTION_GROUPS)
    # so it never produces a detection.
    'Colobus_confuser': 'species/Colobus_confuser',
    # Cercopithecus pogonias, the crowned monkey -- a congener of the target that
    # the species expert identified by ear as the source of the daytime Cernic
    # false positives, the ones that survive the time gate, the isolation filter
    # and the V13 retraining alike. A held-out pogonias recording scores 1.0000
    # on the Cernic output in 23 of 31 windows, so this is a closed-set failure
    # and not a threshold failure: a classifier with no class for a congeneric
    # species has nowhere to put its calls except the class they most resemble.
    # The fix is a class. It began as a confuser folded into Background and is
    # now a detection target in its own right: see DETECTION_GROUPS below, and
    # the measurement that decided it. C. pogonias and C. nictitans are both
    # Near Threatened congeners occurring in Gabon, so a detector for the pair
    # is worth more than a detector for one plus a discarded class, and giving
    # pogonias its own group also scored better for Cernic than folding it away
    # (precision 0.9556 against 0.9530, calls retained 0.8687 against 0.8657).
    # Provenance: cut from a sound library, not from the deployment, confirmed by
    # the species expert who supplied them and consistent with fingerprint
    # matching, which found no match against any clip this repository has
    # exported. They therefore belong to no station and possible_stations()
    # returns an empty list, the same encoding the 172 archival Colobus_guereza
    # clips already carry. That is correct rather than a gap, but it does mean
    # this class treats a field confusion with archival audio, which is the
    # archival-to-field gap the Colobus class already suffers from. Whether the
    # class earns its place is measured by --drop-pogonias, not assumed.
    'C_pogonias': 'species/C_pogonias',
}

# Background noise folders (will be combined into single "Background" class)
# The last entry accumulates hard negatives from the auto-cleanup loop.
# To add impulsive-noise negatives (gunshots, branch-snaps) in a later round,
# create a 'background/impulsive_noise' folder and uncomment the line below.
BACKGROUND_FOLDERS = [
    'background/background noise Clips 5sec',
    'background/Cercocebus torquatus Clips 5s',
    'background/wrong classified',
    'background/Pan troglodytes Clips 5sec',
    # 'background/impulsive_noise',
    'outputs/auto_cleanup/auto_flagged_fp',
    # Confirmed field false positives mined from dev stations (IPA1-18) with
    # scripts/mine_field_negatives.py. Distribution-matched hard negatives (real
    # forest birds/insects/sawing/speech recorded by the same AudioMoth). Label
    # safety: all Colobus clips are taken (no real Colobus at any dev station),
    # but Cernic clips are YAMNet-gated -- only kept when an independent tagger
    # calls the window bird/insect/etc., so real putty-nose calls are never
    # pulled into Background. The held-out test stations IPA19/20 never feed in.
    # Safe to list before it exists -- scan_audio_files() warns and skips a
    # missing folder.
    'background/field_fp_negatives',
    # Windows drawn uniformly at random from the deployment recordings, one
    # per-station subfolder so possible_stations() still holds a station out.
    #
    # Everything else in this list was chosen by something: a detector fired on
    # it, a reviewer judged it, or a person curated it. Measured on the manifest
    # before these were added, 93 % of Background came from the model's own false
    # positives or from reviewed detections, and even the 17 101 BirdNET clips
    # were picked by a detector rather than drawn from the audio. That teaches
    # the model to separate calls from THINGS A DETECTOR ALREADY REACTED TO,
    # which is enough to re-rank a candidate list and not enough to scan a
    # recording: a held-out model turned loose on IPA4ST produced 1 154
    # detections and the first six checked by ear were all wrong, at confidences
    # up to 0.988.
    #
    # These are what the forest actually sounds like. Screened with the deployed
    # grouped-argmax rule: 0.7 % of the draws would have produced a detection and
    # are held back in data/outputs/random_mine_suspect for a person.
    #
    # WITHDRAWN until that screen is validated. The screen's reliability equals
    # the model's recall, which is the one quantity this project has never
    # measured, so labelling 19 573 clips Background on its say-so is circular:
    # a call the model misses becomes a negative example teaching it to keep
    # missing calls. 200 of the not-fired clips are in
    # data/outputs/random_notfired_sample awaiting a human ear
    # (data/outputs/LISTEN_3_notfired.csv). Re-enable this line when that comes
    # back clean; the sweep that used it is data/outputs/v13_loso_randombg.csv
    # and its numbers should be treated as provisional until then.
    # 'background/random_forest',
]

# Subtrees that must NOT be swept into Background even though they sit inside a
# folder listed above. scan_audio_files() skips these and says so.
#
# WHY THIS EXISTS. 'outputs/auto_cleanup/auto_flagged_fp' holds 4 344 clips in
# two kinds of subfolder. The station-named ones (ipa2st_cernic_bulk_birds,
# ipa14st_v4_cernic_fp, ...) were mined per station and hand-checked, and
# cross-matching them against the 6 189 reviewed detections turns up no
# contradiction at all -- they are good hard negatives and stay.
#
# The two named after *filters* are different. Nothing decided their contents
# except the filter itself, and both audit badly:
#
#   mahal/    386 clips, 129 reachable by the review, 44 of 44 listened were
#             confirmed GENUINE CALLS. Mahalanobis distance flags whatever is
#             far from the training distribution, and a loud unambiguous call
#             is exactly that.
#   yamnet/   658 clips, 20 of 24 listened were genuine. YAMNet is disabled
#             precisely because it flags 51.8 % of real calls.
#
# Loading these as Background trains the model to reject the calls it is most
# confident about -- 20 of them come from IPA19/IPA20, the two stations the
# configuration says are held out. This happened on every previous run, with no
# user action required, because the parent folder is listed above and
# scan_audio_files() walks it recursively. Do not re-enable them without
# labelling the 657 unlabelled clips first (data/outputs/unlabelled_dumps).
BACKGROUND_EXCLUDE = [
    'outputs/auto_cleanup/auto_flagged_fp/mahal',
    'outputs/auto_cleanup/auto_flagged_fp/yamnet',
]

# AUDIO PARAMETERS
SAMPLE_RATE = 44100  # Hz
CLIP_DURATION = 2.0  # seconds — length of every TRAINING clip

# Call-to-background level, in dB, when a short clip is embedded in real ambient
# (data_loader.embed_in_background). Measured rather than chosen. In the roar
# band against the 2-8 kHz soundscape band, the nine field-verified C. guereza
# clips sit at a median of -1.0 dB with an interquartile range of -4.5 to +3.1.
# The original (3, 15) put every training example at +3.7 dB median, so the
# easiest field clip was harder than a typical training clip and the class was
# never shown the case it has to handle. This range reproduces the field
# distribution and extends past its hard quartile, which is the direction to err
# in: -1.5 dB median, -6.1 to +6.4.
#
# It applies to every class that has short clips, which includes the C. nictitans
# syllables. That class also has 2 535 field-verified calls carrying real field
# levels, so it is far less exposed to this than Colobus, which has none. Whether
# the change costs Cernic anything is visible in the leave-one-station-out sweep.
EMBED_SNR_DB_RANGE = (-6.0, 9.0)
# SLIDING-WINDOW DETECTION (preprocessing.extract_sliding_windows)
# A long field recording is sliced into fixed-length windows that are each
# classified independently, then high-confidence runs are merged into one
# detection (see detection.detect_in_long_audio).
#   WINDOW_SIZE   = length of each window, in seconds. Kept identical to
#                   CLIP_DURATION so the model sees the same 2 s input
#                   distribution it was trained on.
#   WINDOW_STRIDE = how far the window advances each step. 1.0 s on a 2.0 s
#                   window = 50% overlap, so a call that straddles a window
#                   boundary (e.g. sitting across [0-2s] and [2-4s]) is still
#                   fully captured by the overlapping [1-3s] window instead of
#                   being split in half and missed. Smaller stride = finer time
#                   resolution but more windows (slower); larger stride = faster
#                   but higher risk of clipping a call across the boundary.
WINDOW_SIZE = 2.0  # seconds (one detection window == one training clip length)
WINDOW_STRIDE = 1.0  # seconds (50% overlap so boundary-straddling calls aren't lost)

# MEL-SPECTROGRAM PARAMETERS
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
FMIN = 20  # Hz (minimum frequency)
FMAX = 8000  # Hz (maximum frequency, adjust based on primate vocalizations)

# Target image size for VGG19 (will resize spectrogram to this)
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3

# DATA AUGMENTATION PARAMETERS 
AUGMENTATION_CONFIG = {
    'original': 1,  # Keep 1 original version
    'background_noise_mix': 3,  # Mix with 3 different background samples
    'time_chop': 1,  # 1 time cropping augmentation
    'freq_chop': 1,  # 1 frequency cropping augmentation
    'translate': 1,  # 1 frequency translation augmentation
}

# Background noise mixing parameters
BG_MIX_SNR_RANGE = (-5, 10)  # SNR in dB (signal-to-noise ratio range)

# COLOBUS HIGH-FREQUENCY NUISANCE AUGMENTATION (V12)
# The curated Colobus reference clips carry incidental high-frequency bird/insect
# energy. Because that high-freq content correlates with the Colobus label during
# training, the model learned to fire on high-freq TEXTURE and confused forest
# insects/birds (which sit at 2-5 kHz) with Colobus in the field, even after the
# V11 frequency-position head. To break that spurious correlation, every Colobus
# training clip also yields COLOBUS_HF_AUG_COUNT extra variants whose band ABOVE
# COLOBUS_HF_CUTOFF_HZ is replaced with high-freq content from random background
# clips, while the low-frequency roar (the true, invariant Colobus signature) is
# left untouched. With the high band decorrelated from the label, the model is
# forced to key on the low-frequency roar. Applied ONLY to the class named below
# (Cernic's discriminative energy IS high-freq and must be preserved).
COLOBUS_HF_AUG_CLASS = 'Colobus_guereza'
COLOBUS_HF_CUTOFF_HZ = 1500   # roar lives below this; randomize everything above
COLOBUS_HF_AUG_COUNT = 2      # extra high-freq-randomized variants per Colobus clip

# Geometric augmentation parameters
# Geometric augmentation strength, as a fraction of the spectrogram.
#
# Sun et al. (2022), the study this augmentation scheme is taken from, modify
# each spectrogram "by a random number between 5% and 10% of the size of the
# original spectrogram, as this reflects the approximate range of variation in
# nature". The values here were 0.1-0.3 and +/-20 of 128 mel rows, which is
# 10-30% and 15.6%: one and a half to three times the published range, and past
# the point where the variant is still the same call.
#
# The frequency translation matters most. The whole reason the head carries a
# CoordConv channel is that absolute frequency separates a Colobus roar from a
# bird trill; shifting a training clip 15.6% up or down teaches the model to
# ignore exactly the cue the architecture was built to exploit.
CHOP_RANGE = (0.05, 0.10)          # crop 5-10% from an edge, per Sun et al.
TRANSLATE_RANGE = (-9, 9)          # 7% of 128 mel rows, inside the same range

# TRAIN/VALIDATION SPLIT
VALIDATION_SPLIT = 0.2  # 20% for validation
RANDOM_SEED = 42

# MODEL PARAMETERS
MODEL_NAME = 'VGG19'
PRETRAINED_WEIGHTS = 'imagenet'
# Pooling head applied to the VGG19 feature map before the dense classifier:
#   'gap'        -> GlobalAveragePooling2D (averages away both frequency and
#                   time; the original head)
#   'freq_bands' -> low/mid/high frequency-band pooling (keeps frequency, V6)
#   'temporal'   -> frequency-pool + 1D-conv over time (keeps WHEN energy
#                   occurs; targets the Cernic-vs-insect/sawing confusion, V7)
#   'temporal_freq' -> per-band Conv1D + cross-band Conv1D + BiLSTM (keeps both
#                   WHEN and WHERE energy occurs; the PRODUCTION V10 head)
#   'temporal_freqpos' -> temporal_freq plus an explicit frequency-coordinate
#                   channel (CoordConv) fused into the feature map before the
#                   band split, so each call texture is tagged with its absolute
#                   frequency. Targets the Colobus(low)-vs-bird(high) confusion
#                   that the position-blind band split leaves unresolved (V11/V12)
# Overridable via the PRIMATE_MODEL_POOLING env var so the standard training
# pipeline can switch heads without editing code. The code default is 'gap';
# set PRIMATE_MODEL_POOLING=temporal_freqpos to reproduce the published V12
# model (use temporal_freq for the earlier V10 model).
MODEL_POOLING = os.environ.get('PRIMATE_MODEL_POOLING', 'gap')
FREEZE_BASE_LAYERS = True  # Freeze VGG19 base layers initially
UNFREEZE_LAST_N_BLOCKS = 2  # Stage-2 fine-tuning unfreezes the last 2 blocks
                            # of the block4_conv4-truncated base (block3, block4);
                            # this is the value used for the published V11/V12 model.

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001
DROPOUT_RATE = 0.5

# Early stopping
PATIENCE = 10  # Stop if validation loss doesn't improve for N epochs
MIN_DELTA = 0.001  # Minimum change to qualify as improvement

# DETECTION PARAMETERS
DETECTION_CONFIDENCE_THRESHOLD = 0.4  # Only keep detections above this
NMS_IOU_THRESHOLD = 0.5  # Non-maximum suppression overlap threshold

# LOW-FREQUENCY SPECTRAL-ENERGY GATE (post-processing for Colobus detections)
# A detected Colobus clip is kept only if the fraction of its spectral energy
# below LOWFREQ_GATE_CUTOFF (within the FMIN-FMAX band) is at least
# LOWFREQ_GATE_THRESHOLD. It runs at detection time and adds a `low_freq_ratio`
# column to every Colobus detection, which doubles as a RANKING signal: sorting
# by this ratio surfaces the low-frequency candidates for review and pushes the
# insect false positives (near-zero ratio) to the bottom. No retraining needed.
#
# THRESHOLD HISTORY -- read this before changing the number.
# The original 0.20 was calibrated against the Colobus_confuser clips, whose
# ratios top out at 0.092: against insects and the pulsed high-frequency forest
# sound the gate is close to perfect, and 0.20 sat in a wide empty gap. The
# field then produced a negative population that calibration never saw. Of the
# 253 C. guereza detections the 16-station deployment returned, manual listening
# found no genuine roar at all -- they are thunder and other low-frequency noise.
# Thunder is not a high-frequency intruder a low-frequency test can reject: it is
# itself low-frequency, median ratio 0.396 and upper quartile 0.826, which
# overlaps the reference roars the gate exists to protect.
#
# Recalibrated against that population (scripts/calibrate_colobus_gate.py):
#
#     threshold   reference roars kept   field detections kept
#         0.20                  97.6 %                  89.3 %   <- original
#         0.40                  93.4 %                  49.4 %   <- current
#         0.50                  90.9 %                  41.1 %
#
# 0.40 removes 45 % of the field detections for 4.2 points of reference recall.
# Going further has a poor exchange rate, and the recall side of that trade is
# the side that is not measured (see below).
#
# WHAT THIS NUMBER DOES NOT REST ON. "Roars kept" is measured on the 617
# reference windows, cut from 172 expert-labelled source clips. No field
# detection has ever been confirmed as a genuine roar, so field recall for
# C. guereza is unmeasured, and the gate can only be scored on how much of a
# known-bad population it removes. The gate is a mitigation, not the fix: a
# detector that fires confidently on thunder (median confidence 0.927) has a
# training problem, and the 253 clips are hard negatives for the next model.
#
# A second criterion was searched for and is NOT used. Low-band spectral
# flatness, crest factor, envelope variability and onset rate all separate the
# two populations at AUC 0.46-0.59, i.e. not at all. Envelope modulation in the
# 1-8 Hz pulse band reaches 0.715, real but modest, and buying field rejection
# with it costs reference recall fast (lf 0.40 + pulse 2.38 -> 83.5 % roars kept,
# 32.4 % field kept). With no confirmed field positive to check the recall cost
# against, that trade is not worth making blind.
# Turned off in favour of the out-of-distribution distance below. The evidence
# is the three rounds of expert listening: this gate removed 33 of 35 dawn
# detections at IPA1ST and all 7 at IPA4ST, and every one of the survivors was
# still a microphone knock, so it was not selecting for roars -- it was
# selecting for low frequency, which knocks also have. The measurement that
# settles it is on the nine field-verified roars: at the shipped cutoff this
# gate keeps 93.4 % of reference roars but rejected 100 % of the field Colobus
# detections it was applied to, while the OOD distance keeps every field roar.
# Left in the code and re-enabled by one line, because it is a published result
# and the ablation in the paper depends on being able to reproduce it.
LOWFREQ_GATE_ENABLED = False   # apply the gate inside detect_in_long_audio
LOWFREQ_GATE_CUTOFF = 1500     # Hz
LOWFREQ_GATE_THRESHOLD = 0.40  # recalibrated against the 253 field detections;
                               # keeps 93.4% of reference roars, cuts field
                               # detections from 89.3% to 49.4% (was 0.20)

# OUT-OF-DISTRIBUTION DISTANCE AT DETECTION TIME
# The classifier is closed-set: every window must be assigned one of the five
# classes, and softmax offers no way to answer "none of these". Deployment is
# open-set -- almost everything a sliding window meets belongs to no class the
# model was trained on -- so a microphone knock is forced into the nearest
# cluster and reported at 0.9999. Three rounds of expert listening returned the
# same verdict for that reason, and no threshold on confidence can fix it,
# because the model is not miscalibrated: on held-out reviewed windows it is
# 99.2 % accurate with an expected calibration error of 0.005.
#
# Distance in feature space can express what softmax cannot. Measured on the
# 55 pops the expert rejected against the 9 field-verified roars, a cutoff at
# the 90th percentile of in-sample library distances rejects 98 % of the pops
# and keeps every roar. The equivalent numbers for the low-frequency gate are
# 94 % and, at IPA4ST, zero -- it removed every Colobus detection there,
# because energy below 1.5 kHz is not what separates a roar from a knock.
#
# Annotated always, gated only when asked: the column supports ranking and can
# be applied after the fact, while turning the gate on changes every field
# number in the paper and is therefore an explicit decision.
OOD_DISTANCE_ENABLED = True    # record `ood_distance` on every detection
OOD_GATE_ENABLED = True        # drop detections beyond the cutoff below
OOD_GATE_PERCENTILE = 90       # default: percentile of that class's own
                               # in-sample distances
OOD_FEATURE_LAYER = 'dense_256'
OOD_STATS_CACHE = 'outputs/ood_class_stats.npz'
# One statistics file per head, looked up by fingerprint. A single shared path
# meant scanning a second station loaded the first station's statistics.
OOD_STATS_DIR = 'outputs/ood_stats'

# Per-class percentile overrides.
#
# WHAT THIS WAS FITTED FOR, AND WHY THAT NO LONGER HOLDS (checked 2026-08-30).
#
# Set on 2026-08-10 against the then-current Colobus statistics for the IPA4ST
# head: the nine field-verified roars sat at distances 97 to 321 while the 55
# pops the expert rejected started at 293, so the two distributions touched and
# p90 (202.9) would have rejected two of the nine. The smallest cutoff keeping
# all nine was 321 -- that head's 97th percentile -- and it still rejected 98 %
# of the pops. It was expressed as a percentile rather than as 321 because 321
# is a coordinate in one head's feature space, and every LOSO fold trains its
# own head.
#
# Those statistics are gone. ood_stats/fold_IPA4ST.npz was rewritten under its
# own filename on 2026-08-20 when the class statistics were refitted on the
# 2026-08-19 build. The shipped file gives p90 283.7 and p97 377.8, and scoring
# the same nine roars through it admits ONE of the nine at either percentile:
# on the head this override exists for, it now changes nothing. Across all five
# shipped statistics files the best any percentile reaches is five of nine.
# Recompute with scripts/score_colobus_controls.py; the numbers the manuscript
# prints live in data/outputs/colobus_ood_controls.csv and are checked by
# scripts/verify_manuscript_numbers.py.
#
# The override is kept and disclosed rather than re-tuned. Refitting a cutoff
# each time the statistics move, on nine clips, on the criterion that those nine
# pass, is choosing a parameter by its answer. The failure is the more useful
# result: this percentile is not portable across folds for a class evidenced
# this thinly, and the gate as released has no validated positive control for
# C. guereza. Anyone deploying it for this species should score their own
# controls first.
OOD_GATE_PERCENTILE_BY_CLASS = {
    'Colobus_guereza': 97,
}

# AUTO-CLEANUP FILTERS
# The YAMNet cross-check is off by default: measured against the manual review
# of 6189 field detections it flagged 51.8% of genuine C. nictitans calls but
# only 32.5% of the false positives (lift 0.63), dropping precision from 41.0%
# to 36.1%. The cause is taxonomic rather than a tuning problem -- YAMNet
# assigns putty-nosed calls to the same AudioSet classes as forest noise
# (Cricket, Animal, Owl), so no pruning of the suspicious class set separates
# them, and requiring a minimum confidence did not help at any threshold up to
# 0.8. Mahalanobis + temporal isolation alone raise precision to 45.0% while
# keeping 95.3% of genuine calls. Set this True to re-enable the filter for a
# deployment where the target species is better represented in AudioSet.
USE_YAMNET_FILTER = False

# Stations fail in two ways. Most accumulate scattered false positives, which
# the temporal-isolation rule handles. Occasionally one is overrun by a species
# the model never saw: those detections are numerous and consistent, so they
# are neither isolated nor outliers, and only removing the whole cluster
# reaches them -- which is far too destructive at a station with no invasion.
# With this on, the cleanup decides which case each station is, using no labels
# and no threshold shared between stations (see src/station_regime.py).
#
# OFF by default: the triage does not work on this dataset. It assumes an
# invasion shows up as a dense group separated from the rest of its station by
# a clear gap in clustering tightness. Measured across the 16 field stations,
# the widest such gap was a factor of 1.08 to 1.32 -- and the invaded station
# had the SMALLEST of all, at 1.08, because its intruding species accounts for
# 82% of its detections and so forms the bulk of the distribution rather than a
# group separated from it. No threshold can select it without selecting every
# other station first. The filter is kept, and the diagnosis available through
# station_regime.explain(), because the underlying problem is real and a better
# signal may exist; the gap is not it.
USE_STATION_REGIME_FILTER = False

# ISOLATION FILTER: how many same-species neighbours a detection needs
#
# `flag_isolated` was `n_neighbours == 0` -- verified over all 6 189 reviewed
# detections with zero disagreement. That collapses a count ranging 0..58 into
# one bit and throws away everything the count knows. Graded, and applied after
# TIME_FILTER, it is monotone and it is the strongest signal in the review table
# (scripts/calibrate_cleanup.py):
#
#   n_neighbours 0     384 detections   precision 0.104
#                1     437             0.249
#                2-3   366             0.577
#                4-7   594             0.806
#               >=8   1790             0.930
#
# ROC area 0.880 inside the time window, above the detector's own confidence
# (0.771) and the Mahalanobis distance (0.706), same direction at 16/16
# stations -- it is an unnormalised count, not a distance, so it carries no
# per-station scale that has to be calibrated away. That is why it survives
# leave-one-station-out where every earlier threshold died.
#
# Fitted LOSO the cut lands on 2 or 3 at every fold. Held out, gate + this:
#   ungated 0.410  ->  time gate 0.701  ->  + isolation 0.869 at 89.8 % recall
#
# ORDER MATTERS. On its own `n_neighbours >= 2` reaches only 0.465: the
# nocturnal insect chorus that supplies most false positives is the DENSEST
# material in the recording, not the most isolated. The time gate has to remove
# it first. Set to 1 to restore the old `== 0` behaviour.
ISOLATION_MIN_NEIGHBOURS = 2

# MAHALANOBIS: a ranking signal, not a filter
#
# As a binary decision `flag_mahal` has an ROC area of 0.485 over the 6 189
# reviewed detections -- chance. It flags whatever is far from the training
# distribution, and a loud, close, unambiguous call is exactly that. A blind
# second listener judged 44 of 44 clips it had exported as false positives to be
# GENUINE CALLS (data/labels/disputed_68_labels.csv).
#
# It did real damage: flagged clips fed BACKGROUND_FOLDERS through the Step-5
# loop, so the model was trained to reject the calls it was most confident
# about, including 20 clips from the held-out IPA19/IPA20.
#
# The continuous distance is NOT chance (ROC area 0.763) and stays in the review
# ordering. Only the thresholded filter is switched off here.
USE_MAHALANOBIS_FILTER = False

# TIME FILTER FOR FIELD RECORDINGS
# Coarse, FILE-LEVEL filter (it does NOT trim audio — it only decides which
# whole recordings to process). The recording's start time is parsed from its
# filename (e.g. "S20210225T065943" -> 06:59) and the file is kept only if that
# start time falls within [TIME_FILTER_START, TIME_FILTER_END] (inclusive).
#
# WHEN TO TURN IT ON (time_filter=True in get_ipa_station_files):
#   Production / survey runs. Putty-nose and Colobus call mainly in the early
#   morning, so restricting to the dawn window skips most of the day's audio —
#   far less compute and fewer false positives. Use the SAME window for every
#   station so per-station detection counts are comparable in the paper.
# WHEN TO TURN IT OFF (time_filter=False):
#   Debugging / recovering missed calls / auditing one station's full-day
#   behaviour (e.g. per-station spot-checks). Processes every recording.
#
# Set either bound to None to disable filtering entirely.
#
# RECALIBRATED against the 6 189 reviewed detections (scripts/calibrate_time_gate.py).
# The old 05:30-10:30 window was never measured: it keeps only 40.2 % of the
# confirmed calls, so it discards three fifths of the recall it is meant to
# protect. Swept over every (start, end) pair at >= 95 % recall, 05:00-19:00 is
# the optimum:
#
#   no gate         6 189 detections, 2 535 calls, precision 0.4096
#   05:00-19:00     3 571 detections, 2 503 calls, precision 0.7009
#   dropped         2 618 detections,    32 calls, precision 0.0122
#
# 70.8 % of the false positives removed for 1.3 % of the calls. The dropped mass
# is a nocturnal insect chorus -- hour 03 alone is 1 641 detections and zero
# calls. As a discriminator this binary (AUC 0.847) beats the model's own
# confidence (AUC 0.730).
#
# CAUTION: 32 confirmed calls DO occur at night. Anything the manuscript says
# about diel calling patterns must be measured with the gate OFF, or the result
# is circular -- you would be recovering the window you imposed.
TIME_FILTER_START = "05:00"
TIME_FILTER_END = "19:00"

# IPA STATION CONFIGURATION
# Path to the root containing IPA station folders (IPA1ST, IPA2ST, ...)
IPA_ROOT = os.environ.get(
    "PRIMATE_IPA_ROOT",
    os.path.join(DRIVE_ROOT, "field_recordings"),
)

# OUTPUT PATHS
OUTPUT_ROOT = os.environ.get("PRIMATE_OUTPUT_ROOT", os.path.join(DRIVE_ROOT, "outputs"))
PROCESSED_DATA_DIR = os.path.join(OUTPUT_ROOT, "processed_data")
MODEL_SAVE_DIR = os.path.join(OUTPUT_ROOT, "models")
DETECTION_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "detections")
VISUALIZATION_DIR = os.path.join(OUTPUT_ROOT, "visualizations")

# Create output directories if they don't exist. Wrapped in try/except so that
# importing this module never crashes on read-only filesystems (e.g. CI runners
# inspecting the package without access to the data drive).
for directory in [OUTPUT_ROOT, PROCESSED_DATA_DIR, MODEL_SAVE_DIR,
                  DETECTION_OUTPUT_DIR, VISUALIZATION_DIR]:
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as exc:
        print(f" Warning: could not create output directory {directory}: {exc}")


# DERIVED PARAMETERS
N_CLASSES = len(SPECIES_FOLDERS) + 1  # +1 for Background class
CLASS_NAMES = list(SPECIES_FOLDERS.keys()) + ['Background']

# DETECTION GROUPING
# With the merged Cernic class each model class maps directly to its own
# detection group — no probability aggregation needed at detection time.
DETECTION_GROUPS = {
    'Cernic': 'Cernic',
    'Colobus_guereza': 'Colobus_guereza',
    # The confuser is a trained class but NOT a detection target: route its
    # softmax mass into the Background group so a window the model calls
    # "confuser" is excluded from detections exactly like Background. Crucially
    # this keeps the confuser probability OUT of the Colobus_guereza group, so a
    # real guereza window is no longer inflated by confuser energy.
    'Colobus_confuser': 'Background',
    # C. pogonias is a DETECTION TARGET, not a confuser. It began as the latter,
    # on the reasoning that a window the model calls pogonias must not become a
    # Cernic detection. That reasoning still holds and this mapping still
    # delivers it, because a group of its own keeps pogonias mass out of the
    # Cernic group just as Background did. What changes is that such a window now
    # produces a pogonias detection instead of being discarded.
    #
    # The species expert asked for this: C. pogonias and C. nictitans are both
    # Near Threatened congeners occurring in Gabon, and a detector for the pair
    # is worth more than a detector for one of them plus a discarded class. Note
    # what it costs to report, though. The 6 189 reviewed windows carry
    # C. nictitans verdicts only, so there is no field ground truth for pogonias
    # and no way yet to state its precision the way Table tab:loso states
    # C. nictitans's. Until those detections are reviewed, this configuration can
    # be trained and deployed but its pogonias channel cannot be validated.
    'C_pogonias': 'C_pogonias',
    'Background': 'Background',
}

# Calculate expected number of samples per species after augmentation
AUGMENTATION_MULTIPLIER = sum(AUGMENTATION_CONFIG.values())

# HELPER FUNCTIONS
def print_config_summary():
    """Print a summary of the current configuration"""
    print("PRIMATE VOCALIZATION DETECTION - CONFIGURATION SUMMARY")
    print(f"\n Data Paths:")
    print(f"   Audio Root: {AUDIO_ROOT}")
    print(f"   Long Audio Root: {LONG_AUDIO_ROOT}")
    print(f"   Output Root: {OUTPUT_ROOT}")
    
    print(f"\n Species to Detect ({len(SPECIES_FOLDERS)}):")
    for i, (key, folder) in enumerate(SPECIES_FOLDERS.items(), 1):
        print(f"   {i}. {key} <- {folder}")
    
    print(f"\n Background Sources ({len(BACKGROUND_FOLDERS)}):")
    for i, folder in enumerate(BACKGROUND_FOLDERS, 1):
        print(f"   {i}. {folder}")
    
    print(f"\n Audio Parameters:")
    print(f"   Sample Rate: {SAMPLE_RATE} Hz")
    print(f"   Clip Duration: {CLIP_DURATION}s")
    print(f"   Window Size/Stride: {WINDOW_SIZE}s / {WINDOW_STRIDE}s")
    
    print(f"\n Mel-Spectrogram:")
    print(f"   N_FFT: {N_FFT}, Hop: {HOP_LENGTH}")
    print(f"   Mel Bins: {N_MELS}, Freq Range: {FMIN}-{FMAX} Hz")
    print(f"   Target Image Size: {IMG_HEIGHT}x{IMG_WIDTH}x{IMG_CHANNELS}")
    
    print(f"\n Data Augmentation (Multiplier: {AUGMENTATION_MULTIPLIER}x):")
    for aug_type, count in AUGMENTATION_CONFIG.items():
        print(f"   {aug_type}: {count}")
    
    print(f"\n Model:")
    print(f"   Architecture: {MODEL_NAME}")
    print(f"   Classes: {N_CLASSES} ({', '.join(CLASS_NAMES)})")
    print(f"   Batch Size: {BATCH_SIZE}, Epochs: {EPOCHS}")
    print(f"   Learning Rate: {LEARNING_RATE}, Dropout: {DROPOUT_RATE}")
    
    print(f"\n Detection:")
    print(f"   Confidence Threshold: {DETECTION_CONFIDENCE_THRESHOLD}")
    print(f"   NMS IOU Threshold: {NMS_IOU_THRESHOLD}")
    

if __name__ == "__main__":
    print_config_summary()
