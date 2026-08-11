"""
Build a listening page for labelling detection clips by ear.

Why this exists
---------------
The hard-negative loop in the README ("move the flagged clips into a background
folder, retrain, repeat 3-5 times") has no ground truth in it. Whatever the three
cleanup filters flagged became a training negative, and nobody listened. Two of
the folders that fed it -- ``auto_flagged_fp/mahal`` and ``auto_flagged_fp/yamnet``
-- turn out to be full of genuine calls: of the 68 that both the review and a
second listener reached, **64 are real**, and of the 44 the Mahalanobis filter
contributed, **all 44 are real**. Those clips spent several training rounds
teaching the model not to fire on the calls it was most certain about.

Two modes
---------
``--standalone`` writes **one HTML file** with every clip's audio and spectrogram
embedded. That is the mode to use when the labelling is done by someone else: a
``file://`` link resolves on the *recipient's* machine, so sending a path to a
collaborator sends them a path to a file they do not have. A single file can be
put in Drive, downloaded, and double-clicked, and it works offline. Audio is
re-encoded to Ogg Vorbis at 22.05 kHz for this -- the analysis band tops out at
8 kHz, so nothing audible to the task is lost, and it takes the page from
hundreds of megabytes to tens.

Without ``--standalone`` the page sits beside the wav files it references, which
is lighter to regenerate and fine for labelling on this machine.

Labels
------
``--labels`` sets the categories. The default is the call/not-a-call question,
but a set of *kinds* is often more useful than a yes/no: knowing the Colobus
detections are thunder rather than merely "not Colobus" says which negatives to
mine and which augmentation to build. Keys 1-9 map to the categories in order.

Not corrupting the labels
-------------------------
- **Nothing about the model is shown** -- no confidence, no filter name, no
  earlier verdict. A listener told "the model was 100 % sure" labels differently.
- **Order is shuffled**, seeded. Consecutive detections often come from one bout
  in one recording, and labelling those in a row invites a run of identical
  judgements.
- **What plays is the padded clip**, with the analysis window drawn on the
  spectrogram, so a call clipped at the window edge is visible rather than
  guessed at.

Usage
-----
    # for someone else, one file to send
    python scripts/make_annotation_tool.py --clips data/outputs/detected_clips/Colobus_guereza \\
        --standalone --out data/outputs/colobus_review \\
        --labels "Thunderstorm,Wood cut,Bird or insect,Rain,Real Colobus,Unknown" \\
        --title "What is this sound? (none of these are confirmed Colobus)"

    # everything the filters flagged that nobody has labelled
    python scripts/make_annotation_tool.py --unlabelled-dumps
"""
import argparse
import base64
import glob
import io
import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = os.path.join(REPO, "data")
REVIEW_TABLE = os.path.join(DATA, "outputs/auto_cleanup/cleanup_vs_review.csv")
FLAGGED_RE = re.compile(r"^Cernic__(.+?)__t(\d+)s__conf([\d.]+)\.wav$")
REVIEW_RE = re.compile(r"^Cernic__(.+?)__(\d+)s__conf([\d.]+)\.wav$")

SHARE_SR = 22050        # analysis band ends at 8 kHz; 22.05 kHz keeps all of it


# ----------------------------------------------------------------- selection

def unlabelled_dump_clips():
    """
    Clips in the mahal/yamnet dumps that no human has ever labelled.

    The review covers only what V12 detected, and the dumps accumulated across
    earlier model versions, so a window an older model flagged and V12 never
    fired on was never exported for anyone to hear. Matching is on
    (recording, second) and ignores confidence, because two model versions score
    the same two seconds differently -- and that key is unambiguous here: no
    (recording, second) pair in the review carries two verdicts.
    """
    review = pd.read_csv(REVIEW_TABLE)
    labelled = set()
    for f in review["file"]:
        m = REVIEW_RE.match(str(f))
        if m:
            labelled.add((m.group(1), int(m.group(2))))

    out = []
    for path in sorted(glob.glob(os.path.join(
            DATA, "outputs/auto_cleanup/auto_flagged_fp", "*", "*.wav"))):
        if os.path.basename(os.path.dirname(path)) not in ("mahal", "yamnet"):
            continue
        m = FLAGGED_RE.match(os.path.basename(path))
        if m and (m.group(1), int(m.group(2))) in labelled:
            continue
        out.append(path)
    return out


# --------------------------------------------------------------- rendering

def spectrogram_bytes(wav_path, window_s=2.0, padding=0.5, standalone=False):
    """A mel-spectrogram with the analysis window marked, as image bytes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import librosa
    import librosa.display
    import config

    y, sr = librosa.load(wav_path, sr=config.SAMPLE_RATE, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS, fmin=config.FMIN, fmax=config.FMAX)
    db = librosa.power_to_db(mel, ref=np.max)

    # Smaller and JPEG when the image has to be carried inside the HTML; a
    # 250-clip page is otherwise tens of megabytes of PNG.
    figsize, dpi = ((7.2, 2.6), 92) if standalone else ((9, 3.4), 110)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    librosa.display.specshow(db, sr=sr, hop_length=config.HOP_LENGTH,
                             x_axis="time", y_axis="mel",
                             fmin=config.FMIN, fmax=config.FMAX,
                             cmap="magma", ax=ax)
    duration = len(y) / sr
    if duration > window_s + 0.05:
        start = min(padding, max(0.0, (duration - window_s) / 2))
        for x in (start, start + window_s):
            ax.axvline(x, color="#4ade80", lw=1.4, ls="--", alpha=0.9)
    ax.set_ylabel("Hz")
    ax.set_xlabel("seconds")
    fig.tight_layout(pad=0.3)

    buf = io.BytesIO()
    if standalone:
        fig.savefig(buf, format="jpeg", facecolor="#111",
                    pil_kwargs={"quality": 78, "optimize": True})
        mime = "image/jpeg"
    else:
        fig.savefig(buf, format="png", facecolor="#111")
        mime = "image/png"
    plt.close(fig)
    return buf.getvalue(), mime


def audio_bytes(wav_path):
    """The clip as Ogg Vorbis at SHARE_SR, for embedding."""
    import librosa
    import soundfile as sf

    y, _sr = librosa.load(wav_path, sr=SHARE_SR, mono=True)
    buf = io.BytesIO()
    try:
        sf.write(buf, y, SHARE_SR, format="OGG", subtype="VORBIS")
        return buf.getvalue(), "audio/ogg"
    except Exception:
        buf = io.BytesIO()
        sf.write(buf, y, SHARE_SR, format="WAV", subtype="PCM_16")
        return buf.getvalue(), "audio/wav"


def data_uri(payload, mime):
    return f"data:{mime};base64," + base64.b64encode(payload).decode("ascii")


# ---------------------------------------------------------------- the page

PAGE = r"""<meta charset="utf-8">
<title>__TITLE__</title>
<style>
 :root{color-scheme:dark}
 *{box-sizing:border-box}
 body{margin:0;background:#0d0f12;color:#e6e8eb;
      font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
 header{padding:12px 20px;border-bottom:1px solid #23262b;display:flex;
        align-items:center;gap:14px;flex-wrap:wrap;position:sticky;top:0;
        background:#0d0f12;z-index:5}
 h1{font-size:15px;margin:0;font-weight:600}
 .bar{flex:1;min-width:150px;height:6px;background:#23262b;border-radius:4px;
      overflow:hidden}
 .bar i{display:block;height:100%;background:#4ade80;width:0;transition:.2s}
 .tally{font-variant-numeric:tabular-nums;color:#9aa3ad;font-size:13px}
 main{max-width:940px;margin:0 auto;padding:20px}
 .card{background:#14171c;border:1px solid #23262b;border-radius:12px;padding:16px}
 .card.done{border-color:#2f6b45}
 .pos{color:#9aa3ad;font-size:13px;font-variant-numeric:tabular-nums;
      display:flex;justify-content:space-between;margin-bottom:10px}
 img{width:100%;border-radius:8px;display:block;background:#111}
 audio{width:100%;margin-top:10px}
 #labels{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
         gap:9px;margin-top:14px}
 button{padding:12px 10px;border-radius:9px;border:1px solid #2c3037;
        background:#1b1f25;color:#e6e8eb;font-size:15px;cursor:pointer;
        font-weight:500;font-family:inherit}
 button:hover{border-color:#4b5563}
 button.on{background:#1d4ed8;border-color:#1d4ed8}
 .tools{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap;align-items:center}
 .tools button{padding:8px 13px;font-size:13.5px}
 label.gain{display:flex;align-items:center;gap:8px;color:#9aa3ad;font-size:13px;
            margin-left:auto}
 input[type=range]{width:110px}
 textarea{width:100%;margin-top:10px;background:#0f1216;color:#e6e8eb;
          border:1px solid #2c3037;border-radius:8px;padding:9px;font:inherit;
          font-size:14px;resize:vertical;min-height:42px}
 .hint{color:#6b7280;font-size:13px;margin-top:12px}
 kbd{background:#23262b;border:1px solid #343841;border-radius:4px;
     padding:1px 6px;font-size:12px;font-family:ui-monospace,monospace}
 #export{background:#1d4ed8;border-color:#1d4ed8;font-weight:600;
         padding:8px 15px;font-size:13.5px}
 input[type=file]{display:none}
</style>
<header>
  <h1>__TITLE__</h1>
  <div class="bar"><i id="pbar"></i></div>
  <div class="tally" id="tally"></div>
  <button type="button" onclick="document.getElementById('importer').click()"
          style="padding:8px 13px;font-size:13.5px">Import CSV</button>
  <input type="file" id="importer" accept=".csv">
  <button id="export">Export CSV</button>
</header>
<main>
  <div class="card" id="card">
    <div class="pos"><span id="pos"></span><span id="state"></span></div>
    <img id="spec" alt="spectrogram">
    <audio id="audio" controls preload="auto"></audio>
    <div id="labels"></div>
    <textarea id="note" placeholder="Optional note…"></textarea>
    <div class="tools">
      <button data-nav="-1">← Prev</button>
      <button data-nav="1">Next →</button>
      <button id="jump">Next unlabelled</button>
      <button id="loop">Loop: off</button>
      <button id="auto">Auto-advance: on</button>
      <label class="gain">Gain <input type="range" id="gain" min="1" max="12"
             step="0.5" value="1"><span id="gainv">1.0×</span></label>
    </div>
    <div class="hint" id="hint"></div>
  </div>
</main>
<script>
const CLIPS = __CLIPS__, LABELS = __LABELS__, KEY = "annot_" + __KEY__;
let store = JSON.parse(localStorage.getItem(KEY) || "{}");
let i = 0, loop = false, auto = true, gainNode = null, ctx = null;

const $ = s => document.querySelector(s);
const save = () => localStorage.setItem(KEY, JSON.stringify(store));
const labelOf = id => (store[id] || {}).label || "";

LABELS.forEach((name, k) => {
  const b = document.createElement("button");
  b.innerHTML = `<kbd>${k+1}</kbd> &nbsp;${name}`;
  b.dataset.v = name;
  b.onclick = () => mark(name);
  $("#labels").appendChild(b);
});
$("#hint").innerHTML =
  LABELS.map((n,k) => `<kbd>${k+1}</kbd> ${n}`).join(" &nbsp; ")
  + ` &nbsp; <kbd>space</kbd> replay &nbsp; <kbd>←</kbd> <kbd>→</kbd> move`
  + ` &nbsp; <kbd>L</kbd> loop<br>The green dashes mark the 2 s window the`
  + ` detector classified; the rest is context. Answers are saved in this`
  + ` browser as you go — you can close the tab and come back. Press Export`
  + ` when done.`;

function audioChain(){
  if (ctx) return;
  ctx = new (window.AudioContext || window.webkitAudioContext)();
  const src = ctx.createMediaElementSource($("#audio"));
  gainNode = ctx.createGain();
  src.connect(gainNode); gainNode.connect(ctx.destination);
}
function render(){
  const c = CLIPS[i];
  $("#pos").textContent = `${i+1} / ${CLIPS.length}`;
  $("#state").textContent = labelOf(c.id) ? labelOf(c.id) : "not yet labelled";
  $("#spec").src = c.png;
  $("#audio").src = c.wav;
  $("#audio").loop = loop;
  $("#note").value = (store[c.id] || {}).note || "";
  $("#card").classList.toggle("done", !!labelOf(c.id));
  document.querySelectorAll("#labels button").forEach(b =>
    b.classList.toggle("on", labelOf(c.id) === b.dataset.v));
  const n = Object.keys(store).filter(k => store[k].label).length;
  $("#pbar").style.width = (100*n/CLIPS.length) + "%";
  const counts = {};
  Object.values(store).forEach(x => { if(x.label) counts[x.label] = (counts[x.label]||0)+1; });
  $("#tally").textContent = `${n}/${CLIPS.length}  `
    + LABELS.filter(l => counts[l]).map(l => `${l} ${counts[l]}`).join(" · ");
  $("#audio").play().catch(()=>{});
}
const go = d => { i = Math.max(0, Math.min(CLIPS.length-1, i+d)); render(); };
function mark(v){
  const id = CLIPS[i].id;
  store[id] = Object.assign({}, store[id], {label:v, note:$("#note").value});
  save();
  if (auto && i < CLIPS.length-1) go(1); else render();
}
document.querySelectorAll("button[data-nav]").forEach(b =>
  b.onclick = () => go(+b.dataset.nav));
$("#note").onchange = () => {
  const id = CLIPS[i].id;
  store[id] = Object.assign({}, store[id], {note:$("#note").value}); save();
};
$("#jump").onclick = () => {
  const k = CLIPS.findIndex(c => !labelOf(c.id));
  if (k >= 0){ i = k; render(); } else alert("Every clip is labelled.");
};
$("#loop").onclick = e => { loop = !loop; $("#audio").loop = loop;
  e.target.textContent = "Loop: " + (loop ? "on" : "off"); };
$("#auto").onclick = e => { auto = !auto;
  e.target.textContent = "Auto-advance: " + (auto ? "on" : "off"); };
$("#gain").oninput = e => {
  audioChain(); const g = +e.target.value;
  gainNode.gain.value = g; $("#gainv").textContent = g.toFixed(1) + "×";
};
addEventListener("keydown", e => {
  if (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT") return;
  const k = parseInt(e.key, 10);
  if (k >= 1 && k <= LABELS.length) { e.preventDefault(); mark(LABELS[k-1]); }
  else if (e.key === "ArrowRight"){ e.preventDefault(); go(1); }
  else if (e.key === "ArrowLeft") { e.preventDefault(); go(-1); }
  else if (e.key === " ")         { e.preventDefault();
                                    $("#audio").currentTime = 0; $("#audio").play(); }
  else if (e.key.toLowerCase() === "l"){ $("#loop").click(); }
});
$("#export").onclick = () => {
  const esc = x => `"${String(x).replace(/"/g,'""')}"`;
  const rows = [["clip_id","label","note"]].concat(CLIPS.map(c =>
    [c.id, labelOf(c.id), (store[c.id]||{}).note || ""]));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([rows.map(r => r.map(esc).join(",")).join("\n")],
                                        {type:"text/csv"}));
  a.download = __KEY__ + "_labels.csv";
  a.click();
};
$("#importer").onchange = ev => {
  const f = ev.target.files[0]; if (!f) return;
  const r = new FileReader();
  r.onload = () => {
    let added = 0;
    r.result.split(/\r?\n/).slice(1).forEach(line => {
      const m = line.match(/^"([^"]*)","([^"]*)"(?:,"([\s\S]*)")?$/);
      if (m && m[2]) { store[m[1]] = {label:m[2], note:m[3]||""}; added++; }
    });
    save(); render(); alert(`Imported ${added} labels.`);
  };
  r.readAsText(f);
};
render();
</script>
"""


def build(clip_paths, out_dir, title, labels, seed, standalone):
    os.makedirs(out_dir, exist_ok=True)
    spec_dir = os.path.join(out_dir, "spectrograms")
    if not standalone:
        os.makedirs(spec_dir, exist_ok=True)

    print(f"Preparing {len(clip_paths)} clips"
          f"{' (standalone: audio and images inline)' if standalone else ''}")
    entries, rows, nbytes = [], [], 0
    for k, src in enumerate(clip_paths, 1):
        clip_id = os.path.splitext(os.path.basename(src))[0]
        img, mime = spectrogram_bytes(src, standalone=standalone)

        if standalone:
            snd, amime = audio_bytes(src)
            nbytes += len(img) + len(snd)
            entries.append({"id": clip_id,
                            "wav": data_uri(snd, amime),
                            "png": data_uri(img, mime)})
        else:
            dst = os.path.join(out_dir, clip_id + ".wav")
            if os.path.abspath(dst) != os.path.abspath(src):
                shutil.copy(src, dst)
            png = os.path.join(spec_dir, clip_id + ".png")
            with open(png, "wb") as fh:
                fh.write(img)
            entries.append({"id": clip_id, "wav": clip_id + ".wav",
                            "png": "spectrograms/" + clip_id + ".png"})
        rows.append({"clip_id": clip_id,
                     "source_path": os.path.relpath(src, REPO)})
        if k % 25 == 0 or k == len(clip_paths):
            print(f"\r  {k}/{len(clip_paths)}"
                  + (f"  {nbytes/1e6:.0f} MB embedded" if standalone else ""),
                  end="", flush=True)
    print()

    # Seeded shuffle: a run of clips from one bout invites a run of identical
    # judgements. Seeded so regenerating keeps partial work aligned.
    rng = np.random.default_rng(seed)
    rng.shuffle(entries)

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "batch.csv"), index=False)
    key = os.path.basename(out_dir.rstrip("/"))
    page = (PAGE.replace("__CLIPS__", json.dumps(entries, ensure_ascii=False))
                .replace("__LABELS__", json.dumps(labels, ensure_ascii=False))
                .replace("__KEY__", json.dumps(key))
                .replace("__TITLE__", title))
    name = "annotate_standalone.html" if standalone else "annotate.html"
    html = os.path.join(out_dir, name)
    # encoding is explicit: on Windows the default is cp1252, which cannot
    # encode the arrow glyphs in the key legend, so the write raised after all
    # 173 clips had been decoded, resampled, spectrogrammed and embedded --
    # several minutes of work discarded at the last statement. The page also
    # declares utf-8 in its own <meta charset>, so writing it as anything else
    # produced a file that disagreed with itself.
    with open(html, "w", encoding="utf-8") as fh:
        fh.write(page)

    size = os.path.getsize(html)
    print(f"\nWrote {html}  ({size/1e6:.1f} MB)")
    print(f"      {os.path.join(out_dir, 'batch.csv')}  "
          f"(clip_id -> source clip, for feeding labels back)")
    print(f"Labels: {', '.join(labels)}")
    if standalone:
        print("\nThis one file is everything. Put it in Drive or send it "
              "directly;\nthe recipient downloads it and double-clicks. Do not "
              "send a file:// path --\nthat resolves on their machine, where "
              "these clips do not exist.")
    return html


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--clips", help="Folder of wav files to label.")
    src.add_argument("--unlabelled-dumps", action="store_true",
                     help="Every clip in the mahal/yamnet dumps that no human "
                          "has labelled.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pattern", default="**/*.wav")
    ap.add_argument("--labels", default="Real call,Not a call,Unsure",
                    help="Comma-separated categories; keys 1-9 in order.")
    ap.add_argument("--title", default="Is this a real Cercopithecus nictitans call?")
    ap.add_argument("--standalone", action="store_true",
                    help="Embed all audio and images in one HTML file, for "
                         "sending to someone else.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--parts", type=int, default=1,
                    help="Split into N pages. 657 clips in one sitting is a "
                         "lot to ask, and a 40 MB file will not go by email; "
                         "three parts are ~13 MB each and each one is a "
                         "finishable session with its own progress bar.")
    args = ap.parse_args()

    if args.unlabelled_dumps:
        paths = unlabelled_dump_clips()
        out = args.out or os.path.join(DATA, "outputs/unlabelled_dumps")
    else:
        clips_dir = os.path.abspath(args.clips)
        preferred = sorted(glob.glob(os.path.join(
            clips_dir, "*B-what-Santi-reviewed-3s.wav")))
        paths = preferred or sorted(glob.glob(
            os.path.join(clips_dir, args.pattern), recursive=True))
        out = args.out or clips_dir

    if not paths:
        sys.exit("no clips selected")
    if args.limit:
        paths = paths[:args.limit]
    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    if len(labels) > 9:
        sys.exit("at most 9 labels (keys 1-9)")
    out = os.path.abspath(out)

    if args.parts <= 1:
        build(paths, out, args.title, labels, args.seed, args.standalone)
        return

    # Shuffle before splitting, so each part is a fair sample rather than one
    # station's worth: a part that is all one recording would be labelled
    # differently from one that mixes them, and partial completion should still
    # give an unbiased picture.
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(paths))
    chunks = np.array_split(order, args.parts)
    for k, chunk in enumerate(chunks, 1):
        build([paths[i] for i in chunk],
              os.path.join(out, f"part{k}_of_{args.parts}"),
              f"{args.title}  [part {k} of {args.parts}]",
              labels, args.seed, args.standalone)
        print()


if __name__ == "__main__":
    main()
