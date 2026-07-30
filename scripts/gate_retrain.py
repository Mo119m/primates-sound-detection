"""
Test a retrained model on the reviewed clips before spending a full run on it.

Detection over every continuous recording is the expensive step. It does not
have to be run to find out whether a retrained model is better, because the
manual review left behind 6 189 exported clips whose true labels are known. A
candidate scored on those clips answers, in minutes, whether it stops firing on
the known false positives and still fires on the known genuine calls.

The staged protocol
-------------------
    Stage 1  score the candidate on the reviewed clips        (minutes)
    Stage 2  check the gain holds at stations whose clips
             were NOT mined for hard negatives                (same run)
    Stage 3  pilot detection on a sample of continuous audio  (~1 hour)
    Stage 4  the full run                                     (only if 1-3 pass)

Stages 1 and 2 are this script. Stage 3 is scripts/recall_sample.py, whose
sampler picks the audio to run over. Stage 4 is the normal pipeline.

Stage 2 is not optional. Hard-negative mining has exactly the shape of every
improvement that failed on this dataset: negatives mined at one station can
teach the model that station's noise and nothing transferable. The script
therefore asks which stations the negatives came from and reports the gain
separately for the rest.

Usage:
    # predict with a candidate model over the exported clips, then gate
    python scripts/gate_retrain.py \\
        --reviewed data/outputs/auto_cleanup/cleanup_vs_review.csv \\
        --clips-dir data/outputs/detection_clips \\
        --model models/v13_candidate.keras \\
        --mined-from IPA4ST \\
        --out data/outputs/gate_v13

    # or gate predictions produced elsewhere (a CSV with recording, start_s,
    # pred_species, pred_confidence)
    python scripts/gate_retrain.py \\
        --reviewed ... --predictions preds_v13.csv --mined-from IPA4ST
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd  # noqa: E402
import field_gate  # noqa: E402


def predict_over_clips(model_path, clips_dir, reviewed, batch_size=32,
                       verbose=True):
    """
    Run a candidate model over the exported detection clips.

    Imports TensorFlow lazily so the rest of this script -- and the whole gate --
    stays usable in an environment without it, including for scoring predictions
    that were produced on another machine.
    """
    import numpy as np
    import auto_cleanup
    import config
    import detection

    if verbose:
        print(f"Loading {model_path} ...")
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path, compile=False)

    clips = auto_cleanup.load_clips_from_dir(reviewed, clips_dir)
    if not len(clips):
        sys.exit(f"No clips matched under {clips_dir}. Check --clips-dir.")
    if verbose:
        print(f"Loaded {len(clips)} clips; predicting ...")

    X = auto_cleanup.clips_to_model_input(clips)
    probs = model.predict(X, batch_size=batch_size, verbose=1 if verbose else 0)

    labels, group_idx = detection.get_detection_groups()
    rows = []
    for i, r in enumerate(reviewed.itertuples()):
        if i >= len(probs):
            break
        grouped = detection.group_probabilities(probs[i], labels, group_idx)
        top = int(np.argmax(grouped))
        rows.append({"recording": getattr(r, "recording", ""),
                     "start_s": getattr(r, "start_s", None),
                     "pred_species": labels[top],
                     "pred_confidence": float(grouped[top])})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reviewed", required=True,
                    help="cleanup_vs_review.csv -- the reviewed detections with "
                         "their verdicts.")
    ap.add_argument("--predictions", default=None,
                    help="CSV with recording, start_s, pred_species, "
                         "pred_confidence. Omit to predict with --model.")
    ap.add_argument("--model", default=None,
                    help="Candidate model to run over the clips.")
    ap.add_argument("--clips-dir", default=None,
                    help="Folder of exported detection clips (with --model).")
    ap.add_argument("--species", default=None,
                    help="Target species (default: the commonest reviewed).")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Detection confidence threshold to apply to the "
                         "candidate's output (default: none).")
    ap.add_argument("--mined-from", default="",
                    help="Comma-separated stations whose clips were mined as "
                         "hard negatives. The gain is reported separately for "
                         "every other station, and that is what decides the "
                         "verdict.")
    ap.add_argument("--min-call-retention", type=float,
                    default=field_gate.DEFAULT_MIN_CALL_RETENTION)
    ap.add_argument("--min-precision-gain", type=float,
                    default=field_gate.DEFAULT_MIN_PRECISION_GAIN)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    reviewed = pd.read_csv(args.reviewed)
    if "verdict" not in reviewed.columns:
        sys.exit("--reviewed needs a 'verdict' column; pass cleanup_vs_review.csv")
    print(f"Reviewed clips: {len(reviewed)} "
          f"({int((reviewed['verdict'] == 'call').sum())} calls, "
          f"{int((reviewed['verdict'] == 'false_positive').sum())} false positives)")

    if args.predictions:
        preds = pd.read_csv(args.predictions)
    elif args.model and args.clips_dir:
        preds = predict_over_clips(args.model, args.clips_dir, reviewed)
    else:
        ap.error("give --predictions, or both --model and --clips-dir")

    scored, missing = field_gate.apply_candidate(
        reviewed, preds, species=args.species, threshold=args.threshold)
    if missing:
        print(f"\nWARNING: {missing} reviewed clips had no prediction and were "
              f"excluded. If that is most of them, the join key is wrong -- "
              f"check that 'recording' and 'start_s' mean the same thing in "
              f"both files.")

    mined = [s.strip() for s in args.mined_from.split(",") if s.strip()]
    g = field_gate.gate(scored, mined_from=mined,
                        min_call_retention=args.min_call_retention,
                        min_precision_gain=args.min_precision_gain)

    print()
    print(field_gate.summarise_text(g))

    ps = field_gate.per_station(scored)
    if len(ps):
        print("\nPer station (a gain at only one row is the warning sign):")
        print(ps.to_string(index=False))

    lost = field_gate.lost_calls(scored)
    print(f"\n{len(lost)} genuine calls the candidate would no longer emit.")
    if len(lost):
        print("Listen to a few before accepting it: losing faint or truncated "
              "calls may be an acceptable trade, losing ordinary clear ones is "
              "not.")

    out_dir = args.out or os.path.dirname(os.path.abspath(args.reviewed))
    os.makedirs(out_dir, exist_ok=True)
    scored.to_csv(os.path.join(out_dir, "gate_scored.csv"), index=False)
    if len(ps):
        ps.to_csv(os.path.join(out_dir, "gate_per_station.csv"), index=False)
    if len(lost):
        lost.to_csv(os.path.join(out_dir, "gate_lost_calls.csv"), index=False)
    print(f"\nWrote gate_*.csv to {out_dir}/")

    print("\nNext step:")
    if g["verdict"] == "pilot":
        print("  Stage 3 -- pilot detection on a sample of continuous audio:")
        print("    python scripts/recall_sample.py plan --recordings <dir> \\")
        print("        --segments 12 --segment-s 300 --out data/outputs/pilot")
        print("  Run detection over those segments only, and check both that "
              "the new false-positive rate is acceptable and that the calls "
              "annotated there are still found. Only then commit to the full "
              "run.")
    elif g["verdict"] == "mined-only":
        print("  Mine hard negatives from more stations, or keep the candidate "
              "as a local fix for the station it was trained on and say so. Do "
              "not spend the full run on it as a general improvement.")
    else:
        print("  Do not spend the full run on this candidate.")

    sys.exit(0 if g["verdict"] == "pilot" else 1)


if __name__ == "__main__":
    main()
