"""
Build a local listening tool for labelling a set of detection clips.

Written for the clips the auto-cleanup filters sent to Background without anyone
checking them. Cross-matching against the review showed 44 of 129 checkable
``mahal`` clips and 24 of 239 ``yamnet`` clips are genuine calls, but the review
only reaches 368 of those 1044 clips; the other 676 are unlabelled, and a clip
that cannot be labelled has to be thrown away rather than guessed at. This tool
is how they stop being unlabelled.

Design notes, all of them about not corrupting the labels:

- **Nothing about the model is shown.** No confidence, no filter name, no prior
  verdict. A reviewer told "the model was 100 % sure" labels differently, and the
  whole point of this pass is a judgement independent of the model.
- **Spectrogram and audio together.** The spectrogram covers the padded clip with
  the analysis window marked, so a call clipped at the window edge is visible
  rather than guessed at from a two-second excerpt.
- **Ordering is shuffled**, seeded, so a run of clips from one recording does not
  create a run of identical judgements.
- **Answers save as you go** to browser storage, so a half-finished pass survives
  a closed tab, and export is a plain CSV.

Everything is local: the page references the wav files beside it and is opened
with file://. No audio leaves the machine.

Usage:
    python scripts/make_annotation_tool.py --clips data/outputs/disputed_68
    python scripts/make_annotation_tool.py --clips <dir> --title "mahal/yamnet rest"
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


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
    if duration > window_s + 0.05:      # padded clip: mark the analysis window
        start = min(padding, max(0.0, (duration - window_s) / 2))
        for x in (start, start + window_s):
            ax.axvline(x, color="#4ade80", lw=1.4, ls="--", alpha=0.9)
    ax.set_ylabel("Hz")
    ax.set_xlabel("seconds")
    fig.tight_layout(pad=0.3)
    fig.savefig(png_path, facecolor="#111")
    plt.close(fig)


PAGE = """<meta charset="utf-8">
<title>__TITLE__</title>
<style>
 :root{color-scheme:dark}
 body{margin:0;background:#0d0f12;color:#e6e8eb;
      font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
 header{padding:14px 22px;border-bottom:1px solid #23262b;display:flex;
        align-items:center;gap:20px;flex-wrap:wrap;position:sticky;top:0;
        background:#0d0f12;z-index:5}
 h1{font-size:16px;margin:0;font-weight:600}
 .bar{flex:1;min-width:180px;height:7px;background:#23262b;border-radius:4px;
      overflow:hidden}
 .bar i{display:block;height:100%;background:#4ade80;width:0;transition:.2s}
 .tally{font-variant-numeric:tabular-nums;color:#9aa3ad;font-size:13px}
 main{max-width:960px;margin:0 auto;padding:22px}
 .card{background:#14171c;border:1px solid #23262b;border-radius:12px;
       padding:18px;margin-bottom:16px}
 img{width:100%;border-radius:8px;display:block;background:#111}
 audio{width:100%;margin-top:12px}
 .row{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px}
 button{flex:1;min-width:130px;padding:13px 10px;border-radius:9px;
        border:1px solid #2c3037;background:#1b1f25;color:#e6e8eb;
        font-size:15px;cursor:pointer;font-weight:500}
 button:hover{border-color:#4b5563}
 button.call{border-color:#166534}    button.call.on{background:#166534}
 button.no{border-color:#7f1d1d}      button.no.on{background:#7f1d1d}
 button.unsure{border-color:#78350f}  button.unsure.on{background:#78350f}
 .nav{display:flex;gap:10px;margin-top:8px}
 .nav button{flex:0 0 auto;min-width:90px;padding:9px 14px;font-size:14px}
 .hint{color:#6b7280;font-size:13px;margin-top:12px}
 .meta{color:#9aa3ad;font-size:13px;margin-bottom:10px;
       font-variant-numeric:tabular-nums}
 .done{background:#132b1a;border-color:#166534}
 #export{background:#1d4ed8;border-color:#1d4ed8;font-weight:600}
</style>
<header>
  <h1>__TITLE__</h1>
  <div class="bar"><i id="pbar"></i></div>
  <div class="tally" id="tally"></div>
  <button id="export" style="flex:0 0 auto;min-width:150px;padding:9px 16px;
          font-size:14px">导出 CSV</button>
</header>
<main>
  <div class="card" id="card">
    <div class="meta" id="meta"></div>
    <img id="spec" alt="spectrogram">
    <audio id="audio" controls preload="auto"></audio>
    <div class="row">
      <button class="call"   data-v="call">1 &nbsp; 真叫声</button>
      <button class="no"     data-v="not_call">2 &nbsp; 不是</button>
      <button class="unsure" data-v="unsure">3 &nbsp; 不确定</button>
    </div>
    <div class="nav">
      <button data-nav="-1">← 上一个</button>
      <button data-nav="1">下一个 →</button>
      <button id="loop">循环播放: 关</button>
      <button id="jump">跳到第一个未标</button>
    </div>
    <div class="hint">键盘: <b>1</b> 真叫声 &nbsp; <b>2</b> 不是 &nbsp;
      <b>3</b> 不确定 &nbsp; <b>空格</b> 重播 &nbsp; <b>←/→</b> 前后
      &nbsp;&mdash;&nbsp; 标注后自动跳下一个。答案存在浏览器里,关掉标签页不会丢。</div>
  </div>
</main>
<script>
const CLIPS = __CLIPS__;
const KEY = "annot_" + __KEY__;
let answers = JSON.parse(localStorage.getItem(KEY) || "{}");
let i = 0, loop = false;

const $ = s => document.querySelector(s);
const save = () => localStorage.setItem(KEY, JSON.stringify(answers));

function render(){
  const c = CLIPS[i];
  $("#meta").textContent = `${i+1} / ${CLIPS.length}`;
  $("#spec").src = c.png;
  $("#audio").src = c.wav;
  $("#audio").loop = loop;
  $("#card").classList.toggle("done", !!answers[c.id]);
  document.querySelectorAll("button[data-v]").forEach(b =>
    b.classList.toggle("on", answers[c.id] === b.dataset.v));
  const n = Object.keys(answers).length;
  $("#pbar").style.width = (100*n/CLIPS.length) + "%";
  const cnt = v => Object.values(answers).filter(x => x===v).length;
  $("#tally").textContent =
    `${n}/${CLIPS.length} 已标 · 真 ${cnt("call")} · 假 ${cnt("not_call")} · 不确定 ${cnt("unsure")}`;
  $("#audio").play().catch(()=>{});
}
function go(d){ i = Math.max(0, Math.min(CLIPS.length-1, i+d)); render(); }
function mark(v){
  answers[CLIPS[i].id] = v; save();
  if (i < CLIPS.length-1) go(1); else render();
}
document.querySelectorAll("button[data-v]").forEach(b =>
  b.onclick = () => mark(b.dataset.v));
document.querySelectorAll("button[data-nav]").forEach(b =>
  b.onclick = () => go(+b.dataset.nav));
$("#loop").onclick = e => { loop = !loop; $("#audio").loop = loop;
  e.target.textContent = "循环播放: " + (loop ? "开" : "关"); };
$("#jump").onclick = () => {
  const k = CLIPS.findIndex(c => !answers[c.id]);
  if (k >= 0){ i = k; render(); } else alert("全部标完了");
};
addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  const map = {"1":"call","2":"not_call","3":"unsure"};
  if (map[e.key]) { e.preventDefault(); mark(map[e.key]); }
  else if (e.key === "ArrowRight") { e.preventDefault(); go(1); }
  else if (e.key === "ArrowLeft")  { e.preventDefault(); go(-1); }
  else if (e.key === " ")          { e.preventDefault();
                                     $("#audio").currentTime = 0;
                                     $("#audio").play(); }
});
$("#export").onclick = () => {
  const rows = [["clip_id","label"]].concat(
    CLIPS.map(c => [c.id, answers[c.id] || ""]));
  const csv = rows.map(r => r.map(x => `"${x}"`).join(",")).join("\\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], {type:"text/csv"}));
  a.download = __KEY__ + "_labels.csv";
  a.click();
};
render();
</script>
"""


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", required=True,
                    help="Folder of wav files to label.")
    ap.add_argument("--pattern", default="*B-what-Santi-reviewed-3s.wav",
                    help="Which wavs to present (default: the padded review "
                         "copies, which show the call in context).")
    ap.add_argument("--title", default="标注: 这些片段里哪些是真的 Cernic 叫声")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    clips_dir = os.path.abspath(args.clips)
    wavs = sorted(glob.glob(os.path.join(clips_dir, args.pattern)))
    if not wavs:
        wavs = sorted(glob.glob(os.path.join(clips_dir, "*.wav")))
    if not wavs:
        sys.exit(f"no wav files under {clips_dir}")

    spec_dir = os.path.join(clips_dir, "spectrograms")
    os.makedirs(spec_dir, exist_ok=True)

    print(f"Rendering {len(wavs)} spectrograms...")
    entries = []
    for k, w in enumerate(wavs, 1):
        base = os.path.splitext(os.path.basename(w))[0]
        png = os.path.join(spec_dir, base + ".png")
        if not os.path.exists(png):
            render_spectrogram(w, png)
        entries.append({
            "id": base,
            "wav": os.path.basename(w),
            "png": "spectrograms/" + os.path.basename(png),
        })
        if k % 10 == 0 or k == len(wavs):
            print(f"\r  {k}/{len(wavs)}", end="", flush=True)
    print()

    # Shuffle so consecutive clips from one recording do not invite a run of
    # identical judgements. Seeded, so the order is the same if the page is
    # regenerated and half-finished answers still line up.
    rng = np.random.default_rng(args.seed)
    rng.shuffle(entries)

    key = os.path.basename(clips_dir.rstrip("/"))
    page = (PAGE.replace("__CLIPS__", json.dumps(entries, ensure_ascii=False))
                .replace("__KEY__", json.dumps(key))
                .replace("__TITLE__", args.title))
    out = os.path.join(clips_dir, "annotate.html")
    with open(out, "w") as fh:
        fh.write(page)

    print(f"\nWrote {out}")
    print(f"  {len(entries)} clips, shuffled with seed {args.seed}")
    print("  Nothing about the model is shown -- no confidence, no filter name,")
    print("  no earlier verdict -- so the labels stay independent of it.")
    print(f"\n  open {out}")


if __name__ == "__main__":
    main()
