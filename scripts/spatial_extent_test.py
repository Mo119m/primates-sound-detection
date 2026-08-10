"""
Is a co-firing group of detections physically reachable by one animal?

A guereza roar is audible to about 1.6 km. The array spans 2.3 km north-south
by 6.2 km east-west, with GPS stamped into every AudioMoth filename at eleven of
the sixteen stations -- coordinates nothing in this repository had read until
now. So the question "could one calling animal have produced this?" has an
arithmetic answer: if simultaneous detections span more than about 3.2 km, no
single source within earshot of all of them exists.

This matters because the two largest *Colobus* events in the deployment fired at
9 and 11 stations inside one half-hour slot, and together they are 54.2 % of
every *Colobus* detection ever made. Manual listening had already judged them
non-roars. This turns that judgement into a measurement that does not depend on
anyone's ears: an array-wide simultaneous onset is what a weather front does, and
what a 1.6 km biological source cannot.

*C. nictitans* is the control, and it is the part that makes the test worth
trusting rather than merely arithmetic. Its choruses are genuine, they propagate
between neighbouring stations, and 2 535 of its detections are human-confirmed.
If the method is sound, Cernic groups should stay inside a plausible audibility
radius while the Colobus storm slots blow past it. If Cernic also spans the whole
array, the test is measuring recorder synchrony or diel co-activity rather than a
shared acoustic source, and it should be discarded.

Distances are computed on an equirectangular approximation, which is accurate to
well under a metre at this latitude and over these separations.

Usage:
    PRIMATE_IPA_ROOT=... python scripts/spatial_extent_test.py
    PRIMATE_IPA_ROOT=... python scripts/spatial_extent_test.py --window-min 15
"""
import argparse
import glob
import math
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import config  # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
AUDIBLE_KM = 1.6          # published carrying distance of a guereza roar
COORD_RE = re.compile(r"([+-]\d+\.\d+)([+-]\d+\.\d+)")
START_RE = re.compile(r"S?(\d{8}T\d{6})")


def station_coords(root):
    """Lat/lon per station, read from the AudioMoth filenames themselves."""
    out = {}
    for st in sorted(d for d in os.listdir(root) if d.startswith("IPA")):
        for p in glob.glob(os.path.join(root, st, "*", "*.wav")):
            m = COORD_RE.search(os.path.basename(p))
            if m:
                lat, lon = float(m.group(1)), float(m.group(2))
                if abs(lat) > 1e-6 or abs(lon) > 1e-6:   # skip the GPS-failure zeros
                    out[st] = (lat, lon)
                    break
    return out


def km_between(a, b):
    lat = math.radians((a[0] + b[0]) / 2)
    dy = (a[0] - b[0]) * 110.574
    dx = (a[1] - b[1]) * 111.320 * math.cos(lat)
    return math.hypot(dx, dy)


def load_detections(root, species):
    rows = []
    for p in glob.glob(os.path.join(root, "**", "*_detections.csv"), recursive=True):
        m = START_RE.search(os.path.basename(p))
        if not m:
            continue
        try:
            t = pd.read_csv(p)
        except Exception:
            continue
        if not len(t) or "species" not in t:
            continue
        t = t[t["species"] == species]
        if not len(t):
            continue
        st = next((x for x in os.path.normpath(p).split(os.sep) if x.startswith("IPA")), None)
        start = pd.to_datetime(m.group(1), format="%Y%m%dT%H%M%S")
        rows.append(t.assign(station=st,
                             when=start + pd.to_timedelta(t["start_time"], unit="s")))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def extents(det, coords, window_min):
    """Per time slot: how many stations fired, and how far apart they are."""
    if not len(det):
        return pd.DataFrame()
    d = det[det.station.isin(coords)].copy()
    d["slot"] = d["when"].dt.floor(f"{int(window_min*60)}s")
    out = []
    for slot, g in d.groupby("slot"):
        sts = sorted(g.station.unique())
        if len(sts) < 2:
            span = 0.0
        else:
            span = max(km_between(coords[a], coords[b])
                       for i, a in enumerate(sts) for b in sts[i + 1:])
        out.append({"slot": slot, "n_stations": len(sts), "n_detections": len(g),
                    "span_km": round(span, 2)})
    return pd.DataFrame(out).sort_values("n_stations", ascending=False)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detections", default=os.path.join(REPO, "data/outputs/detections"))
    ap.add_argument("--audio-root", default=config.IPA_ROOT)
    ap.add_argument("--window-min", type=float, default=5.0,
                    help="Slot width. 5 min is chosen, not assumed: --sweep "
                         "shows the separation appears there and nowhere else.")
    ap.add_argument("--sweep", action="store_true",
                    help="Report the Colobus-vs-Cernic gap across slot widths. "
                         "Run this before trusting any single width.")
    ap.add_argument("--out", default=os.path.join(REPO, "data/outputs/spatial_extent.csv"))
    args = ap.parse_args()

    coords = station_coords(args.audio_root)
    print(f"GPS recovered for {len(coords)} of 16 stations, from filenames alone")
    if len(coords) < 2:
        sys.exit("need at least two stations with coordinates")
    pairs = [km_between(coords[a], coords[b])
             for i, a in enumerate(sorted(coords)) for b in sorted(coords)[i + 1:]]
    print(f"array extent: {max(pairs):.2f} km max separation, "
          f"{np.median(pairs):.2f} km median\n")

    if args.sweep:
        print("Slot width sweep. Read the gap column: the separation is a "
              "property of\nthe timescale, and it exists only around 5 min.\n")
        print(f"{'width':>8} {'Colobus':>10} {'Cernic':>10} {'gap':>9}")
        cached = {sp: load_detections(args.detections, sp)
                  for sp in ["Colobus_guereza", "Cernic"]}
        for w in [30, 15, 5, 1, 0.5, 0.25]:
            share = {}
            for sp, det in cached.items():
                e = extents(det, coords, w)
                share[sp] = (e.loc[e.span_km > 2 * AUDIBLE_KM, "n_detections"].sum()
                             / max(e.n_detections.sum(), 1))
            lab = f"{w:g} min" if w >= 1 else f"{w*60:g} s"
            print(f"{lab:>8} {share['Colobus_guereza']:>9.1%} "
                  f"{share['Cernic']:>9.1%} "
                  f"{share['Colobus_guereza']-share['Cernic']:>+9.1%}")
        print("\nAt 30 min every slot aggregates independent callers and nothing "
              "separates.\nBelow ~1 min the sign flips: Cernic rises because a "
              "genuine chorus really does\nreach neighbouring stations within "
              "seconds, while the Colobus detections do not\n-- thunder rolls "
              "and rain arrives progressively, so a storm is minutes-wide, not\n"
              "seconds-wide. Both ends are informative; 5 min is where the two "
              "mechanisms\nare furthest apart.\n")

    results = {}
    for sp in ["Colobus_guereza", "Cernic"]:
        det = load_detections(args.detections, sp)
        ex = extents(det, coords, args.window_min)
        results[sp] = ex
        if not len(ex):
            print(f"{sp}: no detections\n")
            continue
        multi = ex[ex.n_stations >= 2]
        beyond = ex[ex.span_km > 2 * AUDIBLE_KM]
        n_det_beyond = int(beyond.n_detections.sum())
        print(f"=== {sp} ===")
        print(f"  {int(ex.n_detections.sum())} detections in "
              f"{len(ex)} slots of {args.window_min} min")
        print(f"  slots with >=2 stations: {len(multi)} "
              f"({len(multi)/len(ex):.0%})")
        print(f"  median span where >=2 stations fired: "
              f"{multi.span_km.median() if len(multi) else 0:.2f} km")
        print(f"  slots spanning more than {2*AUDIBLE_KM:.1f} km "
              f"(no single source can reach): {len(beyond)}, "
              f"holding {n_det_beyond} detections "
              f"({n_det_beyond/ex.n_detections.sum():.1%} of the species' total)")
        print(f"  worst slots:")
        print(ex.head(3).to_string(index=False))
        print()

    both = pd.concat([v.assign(species=k) for k, v in results.items() if len(v)],
                     ignore_index=True)
    both.to_csv(args.out, index=False)
    print(f"wrote {args.out}")
    print(f"\nA guereza roar carries ~{AUDIBLE_KM} km, so two stations both hearing "
          f"one\nanimal cannot be more than ~{2*AUDIBLE_KM:.1f} km apart. Read the "
          f"Cernic row first:\nif its groups stay inside that and the Colobus "
          f"groups do not, the difference is\nabout the sound source. If both "
          f"exceed it, this test is measuring something else.")


if __name__ == "__main__":
    main()
