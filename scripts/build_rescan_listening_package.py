"""Build the expert's listening package from the corrected rescans.

Stratified, not pooled, because the pooled draw would average away the two
things the scans actually found:

- The pogonias classes of different folds contradict each other. IPA2ST rose
  4 -> 24 under the corrected threshold while IPA4ST fell 44 -> 1, and IPA20ST
  produced 9,703 -- a flood that is itself a finding about a 177-clip congener
  class. Pogonias is therefore sampled per station, never pooled.
- IPA19ST returned 49 Colobus candidates shaped like a dawn chorus, and
  IPA20ST another 17. One confirmed roar rewrites the paper's C. guereza
  section, so every Colobus candidate goes in, none sampled away.

Blinding: the clip name carries the species to listen for and nothing else.
Station, date, confidence and time stay in the key file, which is not shared.
Species must be visible -- the task is "is this a real X" -- but which station
produced a clip is exactly the variable the pogonias disagreement rides on, so
it is hidden from the listener.

Seeded draws (20260828) so the sample is reproducible and the paper can say
how it was drawn.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import soundfile as sf

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SCANS = os.path.join(REPO, "data/outputs/detection_review")
RAW = os.environ.get("PRIMATE_IPA_ROOT", "D:/Gabon raw acoustic data National Park")
OUT = "C:/Users/Fudap/OneDrive/Desktop/FOR_SANTI_2026-08-28"
SEED = 20260828
STATIONS = ["IPA2ST", "IPA4ST", "IPA19ST", "IPA20ST"]

# per-station draw sizes; None = take all
PLAN = {
    "Colobus_guereza": {st: None for st in STATIONS},
    "C_pogonias": {"IPA2ST": None, "IPA4ST": None, "IPA19ST": 20, "IPA20ST": 20},
    "Cernic": {st: 15 for st in STATIONS},
}
SHORT = {"Colobus_guereza": "col", "C_pogonias": "pog", "Cernic": "cer"}
QUESTION = {
    "Colobus_guereza": "is a Colobus guereza roar audible, yes or no",
    "C_pogonias": "is this Cercopithecus pogonias, yes or no (if it is nictitans, write nictitans)",
    "Cernic": "is this Cercopithecus nictitans, yes or no",
}


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(os.path.join(OUT, "clips"), exist_ok=True)

    frames = []
    for st in STATIONS:
        f = os.path.join(SCANS, f"{st}_0500-1900_evalfix", f"{st}_summary.csv")
        d = pd.read_csv(f)
        d["station"] = st
        frames.append(d)
    allscans = pd.concat(frames, ignore_index=True)

    # one glob over the raw tree; 196 individual finds would walk it 196 times
    paths = {os.path.basename(p): p
             for p in glob.glob(os.path.join(RAW, "*", "*", "*.wav"))}

    chosen = []
    for sp, per_st in PLAN.items():
        for st, k in per_st.items():
            pool = allscans[(allscans.species == sp) & (allscans.station == st)]
            if len(pool) == 0:
                continue
            take = pool if k is None or len(pool) <= k else \
                pool.sample(n=k, random_state=int(rng.integers(0, 2**31)))
            chosen.append(take)
    sel = pd.concat(chosen, ignore_index=True)
    # shuffle within species so clip order carries no station structure
    sel = sel.sample(frac=1, random_state=SEED).reset_index(drop=True)

    rows, missing = [], 0
    counters = {}
    for _, r in sel.iterrows():
        src_name = str(r.source_file)
        if not src_name.endswith(".wav"):
            src_name += ".wav"
        src = paths.get(src_name)
        if src is None:
            missing += 1
            continue
        i = counters.get(r.species, 0)
        counters[r.species] = i + 1
        name = f"{SHORT[r.species]}{i:03d}.wav"
        info = sf.info(src)
        sr = info.samplerate
        a = max(0, int((float(r.start_time) - 2) * sr))
        b = min(info.frames, int((float(r.start_time) + 4) * sr))
        data, _ = sf.read(src, start=a, stop=b)
        sf.write(os.path.join(OUT, "clips", name), data, sr)
        rows.append({"clip": name, "species": r.species, "station": r.station,
                     "source_file": src_name, "start_s": float(r.start_time),
                     "confidence": float(r.confidence)})

    key = pd.DataFrame(rows)
    # the key stays LOCAL -- it carries stations and confidences
    key.to_csv(os.path.join(REPO, "data/outputs/detection_review",
                            "rescan_package_key_2026-08-28.csv"), index=False)
    sheet = key[["clip", "species"]].copy()
    sheet["verdict"] = ""
    sheet["note"] = ""
    sheet.to_csv(os.path.join(OUT, "VERDICT_SHEET.csv"), index=False)

    with open(os.path.join(OUT, "README_FIRST.txt"), "w", encoding="ascii") as f:
        f.write(
            "DETECTIONS FROM THE RETRAINED MODEL -- 2026-08-28\n\n"
            "Each clip is 6 s: the 2 s detection window plus 2 s before and 4 s\n"
            "after. The name tells you which species the model claimed:\n\n")
        for sp in PLAN:
            n = int((key.species == sp).sum())
            f.write(f"  {SHORT[sp]}***.wav  ({n:3d} clips)  {QUESTION[sp]}\n")
        f.write(
            "\nWrite yes / no (or nictitans where offered) in VERDICT_SHEET.csv.\n"
            "If unsure, write unsure -- never leave a row blank.\n\n"
            "WHY THE col CLIPS MATTER MOST\n"
            "  Our paper currently reports Colobus guereza as never confirmed\n"
            "  at these sites. These are the strongest candidates the retrained\n"
            "  model has ever produced, many shaped like a dawn chorus. A\n"
            "  single confirmed roar changes a whole section of the paper.\n\n"
            "WHY THE pog CLIPS ARE ASKED CAREFULLY\n"
            "  Different stations' models disagree with each other about this\n"
            "  species, so your ear is the referee. If a pog clip is actually\n"
            "  nictitans, writing 'nictitans' is exactly the answer we need.\n\n"
            "This batch is SEPARATE from the earlier package (s0000-s0149 and\n"
            "d000-d116). Those two are still the highest priority, especially\n"
            "the 150 s-clips: the paper's headline number waits on them.\n")

    print(f"  cut {len(rows)} clips ({missing} source recordings not found)")
    for sp, n in key.species.value_counts().items():
        print(f"    {sp:18s} {n}")
    print(f"  package: {OUT}")
    print(f"  key (local only): data/outputs/detection_review/rescan_package_key_2026-08-28.csv")
    return 0 if missing == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
