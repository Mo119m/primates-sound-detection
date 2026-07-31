"""
Build a self-contained listening page for labelling detection clips by ear.

Why this exists
---------------
The hard-negative loop in the README ("move the flagged clips into a background
folder, retrain, repeat 3-5 times") has no ground truth in it. Whatever the three
cleanup filters flagged became a training negative, and nobody listened. Two of
the folders that fed it -- ``auto_flagged_fp/mahal`` and ``auto_flagged_fp/yamnet``
-- turn out to be full of genuine calls: of the 68 that both the review and a
second listener reached, **64 are real**, and of the 44 that the Mahalanobis
filter contributed, **all 44 are real**. Those clips spent several training
rounds teaching the model not to fire on the calls it was most certain about.

Another 657 clips in the same two folders have never been labelled by anyone. A
clip with no label cannot be used and cannot be safely discarded, so the only way
out is to listen to them. This makes that as fast as it can be.

Design choices, all of them about not corrupting the labels
-----------------------------------------------------------
- **The page shows nothing about the model.** No confidence, no filter name, no
  earlier verdict. A listener told "the model was 100 % sure" labels differently,
  and the value of a second pass is precisely that it is independent.
- **Order is shuffled** (seeded, so it is reproducible and partial work still
  lines up). Consecutive detections often come from one bout in one recording,
  and labelling those in a row invites a run of identical judgements.
- **What plays is the padded clip**, with the analysis window drawn on the
  spectrogram. A call clipped at the window edge is then visible rather than
  inferred from two seconds of audio.
- **Answers persist in the browser** as you go, and export to CSV. Nothing is
  uploaded; the page reads the wav files sitting beside it.

Usage
-----
    # label an existing folder of clips
    python scripts/make_annotation_tool.py --clips data/outputs/disputed_68

    # build a batch from the manifest: everything the filters flagged that no
    # human has ever labelled
    python scripts/make_annotation_tool.py --unlabelled-dumps \\
        --out data/outputs/dumps_batch

Then open the ``annotate.html`` it writes, label, and press Export. Feed the CSV
back with ``scripts/apply_human_labels.py``.
"""
import argparse
import glob
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


# ----------------------------------------------------------------- selection

def unlabelled_dump_clips():
    """
    Clips in the mahal/yamnet dumps that no human has ever labelled.

    The review covers only what V12 detected, and the dumps accumulated across
    earlier model versions, so a window an older model flagged and V12 never
    fired on was never exported for anyone to hear. Matching is on
    (recording, second) and ignores confidence, because two different model
    versions score the same two seconds differently -- and that key is
    unambiguous here: no (recording, second) pair in the review carries two
    verdicts.
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
        folder = os.path.basename(os.path.dirname(path))
        if folder not in ("mahal", "yamnet"):
            continue
        m = FLAGGED_RE.match(os.path.basename(path))
        if m and (m.group(1), int(m.group(2))) in labelled:
            continue
        out.append(path)
    return out


# ------------------------------------------------------------- spectrograms

def render_spectrogram(wav_path, png_path, window_s=2.0, padding=0.5):
    """One clip as a mel-spectrogram, with the analysis window marked."""
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

    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=110)
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
    fig.savefig(png_path, facecolor="#111")
    plt.close(fig)


# ---------------------------------------------------------------- the page

PAGE = r"""<meta charset="utf-8">
<title>__TITLE__</title>
<style>
 :root{color-scheme:dark}
 *{box-sizing:border-box}
 body{margin:0;background:#0d0f12;color:#e6e8eb;
      font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
 header{padding:12px 20px;border-bottom:1px solid #23262b;display:flex;
        align-items:center;gap:16px;flex-wrap:wrap;position:sticky;top:0;
        background:#0d0f12;z-index:5}
 h1{font-size:15px;margin:0;font-weight:600;white-space:nowrap}
 .bar{flex:1;min-width:160px;height:6px;background:#23262b;border-radius:4px;
      overflow:hidden}
 .bar i{display:block;height:100%;background:#4ade80;width:0;transition:.2s}
 .tally{font-variant-numeric:tabular-nums;color:#9aa3ad;font-size:13px;
        white-space:nowrap}
 main{max-width:960px;margin:0 auto;padding:20px}
 .card{background:#14171c;border:1px solid #23262b;border-radius:12px;padding:16px}
 .card.done{border-color:#2f6b45}
 .pos{color:#9aa3ad;font-size:13px;font-variant-numeric:tabular-nums;
      display:flex;justify-content:space-between;margin-bottom:10px}
 img{width:100%;border-radius:8px;display:block;background:#111}
 audio{width:100%;margin-top:10px}
 .row{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px}
 button{padding:12px 10px;border-radius:9px;border:1px solid #2c3037;
        background:#1b1f25;color:#e6e8eb;font-size:15px;cursor:pointer;
        font-weight:500;font-family:inherit}
 button:hover{border-color:#4b5563}
 .row button{flex:1;min-width:120px}
 .call{border-color:#166534}   .call.on{background:#166534}
 .no{border-color:#7f1d1d}     .no.on{background:#7f1d1d}
 .unsure{border-color:#78350f} .unsure.on{background:#78350f}
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
 #importLabel{cursor:pointer}
 input[type=file]{display:none}
</style>
<header>
  <h1>__TITLE__</h1>
  <div class="bar"><i id="pbar"></i></div>
  <div class="tally" id="tally"></div>
  <label id="importLabel"><button type="button"
     onclick="document.getElementById('importer').click()">Import CSV</button></label>
  <input type="file" id="importer" accept=".csv">
  <button id="export">Export CSV</button>
</header>
<main>
  <div class="card" id="card">
    <div class="pos"><span id="pos"></span><span id="state"></span></div>
    <img id="spec" alt="spectrogram">
    <audio id="audio" controls preload="auto"></audio>
    <div class="row">
      <button class="call"   data-v="call"><kbd>1</kbd> &nbsp;Real call</button>
      <button class="no"     data-v="not_call"><kbd>2</kbd> &nbsp;Not a call</button>
      <button class="unsure" data-v="unsure"><kbd>3</kbd> &nbsp;Unsure</button>
    </div>
    <textarea id="note" placeholder="Optional note (what you heard, why unsure)…"></textarea>
    <div class="tools">
      <button data-nav="-1">← Prev</button>
      <button data-nav="1">Next →</button>
      <button id="jump">Next unlabelled</button>
      <button id="loop">Loop: off</button>
      <button id="auto">Auto-advance: on</button>
      <label class="gain">Gain <input type="range" id="gain" min="1" max="12"
             step="0.5" value="1"><span id="gainv">1.0×</span></label>
    </div>
    <div class="hint">
      <kbd>1</kbd> real &nbsp; <kbd>2</kbd> not &nbsp; <kbd>3</kbd> unsure &nbsp;
      <kbd>space</kbd> replay &nbsp; <kbd>←</kbd> <kbd>→</kbd> move &nbsp;
      <kbd>L</kbd> loop &nbsp; — the green dashes mark the 2 s window the model
      classified; the rest is context. Answers are saved in this browser as you
      go, and the model's own opinion is deliberately not shown.
    </div>
  </div>
</main>
<script>
const CLIPS = __CLIPS__, KEY = "annot_" + __KEY__;
let store = JSON.parse(localStorage.getItem(KEY) || "{}");
let i = 0, loop = false, auto = true, gainNode = null, ctx = null;

const $ = s => document.querySelector(s);
const save = () => localStorage.setItem(KEY, JSON.stringify(store));
const labelOf = id => (store[id] || {}).label || "";

function audioChain(){
  // WebAudio gain: field recordings are quiet and a faint call at the edge of
  // the window is easy to miss at 1x.
  if (ctx) return;
  ctx = new (window.AudioContext || window.webkitAudioContext)();
  const src = ctx.createMediaElementSource($("#audio"));
  gainNode = ctx.createGain();
  src.connect(gainNode); gainNode.connect(ctx.destination);
}

function render(){
  const c = CLIPS[i];
  $("#pos").textContent = `${i+1} / ${CLIPS.length}`;
  $("#state").textContent = labelOf(c.id) ? "labelled" : "not yet labelled";
  $("#spec").src = c.png;
  $("#audio").src = c.wav;
  $("#audio").loop = loop;
  $("#note").value = (store[c.id] || {}).note || "";
  $("#card").classList.toggle("done", !!labelOf(c.id));
  document.querySelectorAll("button[data-v]").forEach(b =>
    b.classList.toggle("on", labelOf(c.id) === b.dataset.v));

  const n = Object.keys(store).filter(k => store[k].label).length;
  $("#pbar").style.width = (100*n/CLIPS.length) + "%";
  const cnt = v => Object.values(store).filter(x => x.label===v).length;
  $("#tally").textContent = `${n}/${CLIPS.length} · real ${cnt("call")}`
    + ` · not ${cnt("not_call")} · unsure ${cnt("unsure")}`;
  $("#audio").play().catch(()=>{});
}
const go = d => { i = Math.max(0, Math.min(CLIPS.length-1, i+d)); render(); };
function mark(v){
  const id = CLIPS[i].id;
  store[id] = Object.assign({}, store[id], {label:v, note:$("#note").value});
  save();
  if (auto && i < CLIPS.length-1) go(1); else render();
}
document.querySelectorAll("button[data-v]").forEach(b =>
  b.onclick = () => mark(b.dataset.v));
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
  const map = {"1":"call","2":"not_call","3":"unsure"};
  if (map[e.key]) { e.preventDefault(); mark(map[e.key]); }
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
    save(); render();
    alert(`Imported ${added} labels.`);
  };
  r.readAsText(f);
};
render();
</script>
"""


def build(clip_paths, out_dir, title, seed, copy_audio):
    os.makedirs(out_dir, exist_ok=True)
    spec_dir = os.path.join(out_dir, "spectrograms")
    os.makedirs(spec_dir, exist_ok=True)

    print(f"Preparing {len(clip_paths)} clips in {out_dir}")
    entries, rows = [], []
    for k, src in enumerate(clip_paths, 1):
        clip_id = os.path.splitext(os.path.basename(src))[0]
        wav_name = clip_id + ".wav"
        dst = os.path.join(out_dir, wav_name)
        if copy_audio and not os.path.exists(dst):
            shutil.copy(src, dst)
        png = os.path.join(spec_dir, clip_id + ".png")
        if not os.path.exists(png):
            render_spectrogram(dst if copy_audio else src, png)
        entries.append({"id": clip_id, "wav": wav_name,
                        "png": "spectrograms/" + clip_id + ".png"})
        rows.append({"clip_id": clip_id,
                     "source_path": os.path.relpath(src, REPO)})
        if k % 25 == 0 or k == len(clip_paths):
            print(f"\r  {k}/{len(clip_paths)}", end="", flush=True)
    print()

    # Seeded shuffle: a run of clips from one bout invites a run of identical
    # judgements. Seeded so regenerating the page keeps partial work aligned.
    rng = np.random.default_rng(seed)
    rng.shuffle(entries)

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "batch.csv"), index=False)
    key = os.path.basename(out_dir.rstrip("/"))
    page = (PAGE.replace("__CLIPS__", json.dumps(entries, ensure_ascii=False))
                .replace("__KEY__", json.dumps(key))
                .replace("__TITLE__", title))
    html = os.path.join(out_dir, "annotate.html")
    with open(html, "w") as fh:
        fh.write(page)

    print(f"\nWrote {html}")
    print(f"      {os.path.join(out_dir, 'batch.csv')}  "
          f"(clip_id -> source clip, for feeding labels back)")
    print(f"\n  open {html}")
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
    ap.add_argument("--pattern", default="*.wav")
    ap.add_argument("--title", default="Is this a real Cercopithecus nictitans call?")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.unlabelled_dumps:
        paths = unlabelled_dump_clips()
        out = args.out or os.path.join(DATA, "outputs/unlabelled_dumps")
        copy_audio = True
    else:
        clips_dir = os.path.abspath(args.clips)
        preferred = sorted(glob.glob(os.path.join(
            clips_dir, "*B-what-Santi-reviewed-3s.wav")))
        paths = preferred or sorted(glob.glob(
            os.path.join(clips_dir, args.pattern)))
        out = args.out or clips_dir
        copy_audio = out != clips_dir

    if not paths:
        sys.exit("no clips selected")
    if args.limit:
        paths = paths[:args.limit]
    build(paths, out, args.title, args.seed, copy_audio)


if __name__ == "__main__":
    main()
