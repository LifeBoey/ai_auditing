"""
Build an HTML page for a human to review the output of detect_label_errors.py.

Inputs:
    scores CSV             label_error_scores.csv from detect_label_errors.py
    ground-truth CSV       file_name,annotation   (the annotations under audit)
    audio dir              folder containing the audio (searched recursively)
    transcript CSVs        optional, file_name,annotation per model, to show
                           every model's own transcript next to the ROVER one

The page lists every utterance, most suspicious first (by confident_errors, then
rover_wer; changeable in the page). Each card has an audio player, the ground
truth and the ROVER (predicted) transcript with differing words highlighted,
the suspect words, and the scores of every model. The reviewer can mark each
ground truth as correct / wrong / unsure, type a corrected transcript, and
export their verdicts as CSV. Verdicts are kept in the browser between visits.

Usage:
    python review_label_errors.py --scores-csv label_error_scores.csv \\
        --ground-truth-csv ground_truth_transcribe_test.csv \\
        --audio-dir ../../asr_testing/LibriSpeech_250 \\
        --transcripts transcripts/*.csv --output-html label_error_review.html

By default the page links to the audio files by relative path, so keep the HTML
where it was written (or pass --embed-audio to put the audio inside the HTML,
which makes one large but portable file).
"""

import os
import re
import csv
import json
import base64
import argparse
import difflib
import mimetypes
from pathlib import Path

AUDIO_EXTS = {".flac", ".wav", ".mp3", ".ogg", ".m4a", ".opus"}
# Score columns shown on every card, in this order, with a short description.
METRIC_COLUMNS = [
    ("confident_errors", "max(unsupported, missing): words the models agree are wrong"),
    ("unsupported_words", "annotation words no model produced"),
    ("missing_words", "words every model produced that the annotation lacks"),
    ("rover_wer", "WER of annotation vs ROVER transcript"),
    ("rover_cer", "CER of annotation vs ROVER transcript"),
    ("weighted_wer", "WER weighted by model agreement"),
    ("min_model_wer", "WER vs the closest single model"),
    ("median_model_wer", "WER vs the typical model"),
    ("model_disagreement", "mean WER of models vs ROVER (audio difficulty)"),
]
SORT_KEYS = ["confident_errors", "rover_wer", "rover_cer", "weighted_wer",
             "min_model_wer", "median_model_wer", "model_disagreement"]


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def to_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() and "." not in str(value) else number


def find_audio(audio_dir):
    """Map file name -> path for every audio file under audio_dir."""
    found = {}
    for path in Path(audio_dir).rglob("*"):
        if path.suffix.lower() in AUDIO_EXTS:
            found.setdefault(path.name, path)
    return found


def audio_src(path, html_dir, embed):
    if embed:
        mime = mimetypes.guess_type(path.name)[0] or "audio/flac"
        return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"
    try:
        return Path(os.path.relpath(path.resolve(), html_dir)).as_posix()
    except ValueError:  # different drive on Windows
        return path.resolve().as_uri()


def word_key(word):
    return re.sub(r"[^a-z0-9']", "", word.lower())


def word_diff(reference, hypothesis):
    """
    Align two transcripts word by word. Returns (ref_tokens, hyp_tokens), each a
    list of [word, changed] so the page can highlight the words that differ.
    """
    ref, hyp = reference.split(), hypothesis.split()
    matcher = difflib.SequenceMatcher(a=[word_key(w) for w in ref],
                                      b=[word_key(w) for w in hyp], autojunk=False)
    ref_out, hyp_out = [], []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        changed = op != "equal"
        ref_out += [[w, changed] for w in ref[i1:i2]]
        hyp_out += [[w, changed] for w in hyp[j1:j2]]
    return ref_out, hyp_out


def build_items(scores, ground_truth, model_transcripts, audio_files, html_dir, embed):
    items, missing_audio, gt_mismatch = [], 0, 0
    model_names = sorted({c[len("wer_"):] for c in scores[0] if c.startswith("wer_")}
                         | set(model_transcripts))
    for row in scores:
        name = row["file_name"]
        annotation = ground_truth.get(name)
        if annotation is None:
            annotation = row.get("annotation", "")
        elif annotation != row.get("annotation", annotation):
            gt_mismatch += 1
        predicted = row.get("rover_transcript", "")
        gt_tokens, pred_tokens = word_diff(annotation, predicted)

        models = []
        for model in model_names:
            text = model_transcripts.get(model, {}).get(name)
            models.append({
                "name": model,
                "wer": to_number(row.get(f"wer_{model}")),
                "tokens": word_diff(annotation, text)[1] if text is not None else None,
            })

        path = audio_files.get(name)
        if path is None:
            missing_audio += 1
        items.append({
            "file": name,
            "audio": audio_src(path, html_dir, embed) if path else None,
            "gt": gt_tokens,
            "pred": pred_tokens,
            "suspect": row.get("suspect_words", "").split(),
            "flagged": row.get("flagged", "").strip().lower() == "true",
            "metrics": {col: to_number(row.get(col)) for col, _ in METRIC_COLUMNS
                        if col in row},
            "num_models": to_number(row.get("num_models")),
            "models": models,
        })
    return items, missing_audio, gt_mismatch


def build_review_page(scores_csv, ground_truth_csv, audio_dir, output_html,
                      transcript_csvs=(), embed_audio=False):
    """
    Write a self-contained HTML review page for every row of scores_csv.

    Args:
        scores_csv: label_error_scores.csv from detect_label_errors.py.
        ground_truth_csv: CSV (file_name,annotation) of the annotations under audit.
        audio_dir: Folder searched recursively for the audio files.
        output_html: Where the page is written.
        transcript_csvs: Optional per-model transcript CSVs (named by file stem).
        embed_audio: Embed the audio as base64 instead of linking to it.
    """
    scores = read_csv(scores_csv)
    if not scores:
        raise SystemExit(f"{scores_csv} has no rows")
    ground_truth = {r["file_name"]: r["annotation"] for r in read_csv(ground_truth_csv)}
    model_transcripts = {Path(p).stem: {r["file_name"]: r["annotation"] for r in read_csv(p)}
                         for p in transcript_csvs}

    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    items, missing_audio, gt_mismatch = build_items(
        scores, ground_truth, model_transcripts, find_audio(audio_dir),
        output_html.parent.resolve(), embed_audio)

    data = {
        "title": Path(scores_csv).name,
        "storageKey": f"label-review:{Path(scores_csv).resolve()}",
        "metricInfo": [[c, d] for c, d in METRIC_COLUMNS],
        "sortKeys": SORT_KEYS,
        "items": items,
    }
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    output_html.write_text(HTML_TEMPLATE.replace("__DATA__", payload), encoding="utf-8")

    print(f"Wrote {output_html} ({len(items)} utterances, "
          f"{sum(i['flagged'] for i in items)} flagged)")
    if missing_audio:
        print(f"Warning: no audio found under {audio_dir} for {missing_audio} files")
    if gt_mismatch:
        print(f"Warning: {gt_mismatch} annotations in {ground_truth_csv} differ from "
              f"those in {scores_csv}; showing the ground-truth CSV's version")
    unscored = set(ground_truth) - {i["file"] for i in items}
    if unscored:
        print(f"Warning: {len(unscored)} ground-truth files have no scores and are not shown")


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Label Error Review</title>
<style>
:root {
  --bg: #f6f7f9; --card: #ffffff; --text: #1d2330; --muted: #667085; --border: #e3e6eb;
  --accent: #3b5bdb; --bad-bg: #ffe3e3; --bad: #c92a2a; --good-bg: #d3f9d8; --good: #2b8a3e;
  --warn-bg: #fff3bf; --warn: #a66900; --bar: #94a3c4; --bar-hi: #e8590c;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #14161b; --card: #1d2027; --text: #e6e8ec; --muted: #9aa3b2; --border: #2e323b;
    --accent: #7c9cff; --bad-bg: #4a1f22; --bad: #ff8787; --good-bg: #1d3b26; --good: #69db7c;
    --warn-bg: #42371a; --warn: #ffd43b; --bar: #5c6884; --bar-hi: #ff922b;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
header { position: sticky; top: 0; z-index: 5; background: var(--card);
  border-bottom: 1px solid var(--border); padding: 10px 16px; }
.bar { max-width: 1100px; margin: 0 auto; display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: center; }
h1 { font-size: 17px; margin: 0 8px 0 0; }
select, input[type=search], button, textarea { font: inherit; color: var(--text); background: var(--bg);
  border: 1px solid var(--border); border-radius: 6px; padding: 4px 8px; }
button { cursor: pointer; }
button:hover { border-color: var(--accent); }
.stats { color: var(--muted); font-size: 13px; }
.help { color: var(--muted); font-size: 12px; width: 100%; }
kbd { font: 11px ui-monospace, monospace; border: 1px solid var(--border); border-radius: 3px; padding: 0 4px; }
main { max-width: 1100px; margin: 16px auto; padding: 0 16px 80px; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 10px;
  padding: 14px 16px; margin-bottom: 14px; scroll-margin-top: 110px; }
.card.current { outline: 2px solid var(--accent); }
.card.v-gt_ok { border-left: 5px solid var(--good); }
.card.v-gt_wrong { border-left: 5px solid var(--bad); }
.card.v-unsure { border-left: 5px solid var(--warn); }
.top { display: flex; flex-wrap: wrap; gap: 8px 12px; align-items: center; }
.rank { color: var(--muted); font-variant-numeric: tabular-nums; }
.fname { font: 600 14px ui-monospace, monospace; }
.badge { font-size: 12px; border-radius: 999px; padding: 1px 9px; font-weight: 600; }
.badge.flag { background: var(--bad-bg); color: var(--bad); }
.badge.ok { background: var(--good-bg); color: var(--good); }
audio { width: 100%; max-width: 480px; height: 36px; margin-left: auto; }
.noaudio { margin-left: auto; color: var(--bad); font-size: 13px; }
.grid { display: grid; grid-template-columns: 1fr 300px; gap: 16px; margin-top: 10px; }
@media (max-width: 820px) { .grid { grid-template-columns: 1fr; } audio { margin-left: 0; } }
.label { font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); margin: 8px 0 2px; }
.text { font-size: 16px; line-height: 1.7; word-wrap: break-word; }
.text.gt .chg { background: var(--bad-bg); color: var(--bad); border-radius: 3px; padding: 0 2px; }
.text .chg { background: var(--good-bg); color: var(--good); border-radius: 3px; padding: 0 2px; }
.models { font-size: 14px; }
.models .text { font-size: 14px; }
.suspect { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 2px; }
.chip { font: 12px ui-monospace, monospace; border-radius: 4px; padding: 0 5px; }
.chip.plus { background: var(--bad-bg); color: var(--bad); }
.chip.minus { background: var(--good-bg); color: var(--good); }
table { width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }
td { padding: 2px 4px; border-bottom: 1px solid var(--border); }
td.num { text-align: right; white-space: nowrap; }
td.hi { color: var(--bad); font-weight: 700; }
.wbar { height: 7px; background: var(--border); border-radius: 4px; min-width: 60px; }
.wbar span { display: block; height: 100%; border-radius: 4px; background: var(--bar); }
.wbar span.hi { background: var(--bar-hi); }
.review { display: flex; flex-wrap: wrap; gap: 8px; align-items: flex-start; margin-top: 12px;
  border-top: 1px dashed var(--border); padding-top: 10px; }
.review button.on.gt_ok { background: var(--good-bg); color: var(--good); border-color: var(--good); }
.review button.on.gt_wrong { background: var(--bad-bg); color: var(--bad); border-color: var(--bad); }
.review button.on.unsure { background: var(--warn-bg); color: var(--warn); border-color: var(--warn); }
.review textarea { flex: 1 1 320px; min-height: 34px; resize: vertical; }
details summary { cursor: pointer; color: var(--muted); font-size: 13px; margin-top: 6px; }
.empty { text-align: center; color: var(--muted); padding: 40px; }
</style>
</head>
<body>
<header>
  <div class="bar">
    <h1>Label error review</h1>
    <label>Sort <select id="sort"></select></label>
    <label>Show <select id="filter">
      <option value="all">all</option>
      <option value="flagged">flagged only</option>
      <option value="unreviewed">not yet reviewed</option>
      <option value="gt_wrong">marked GT wrong</option>
      <option value="unsure">marked unsure</option>
    </select></label>
    <input type="search" id="search" placeholder="Search file or words">
    <button id="export">Export verdicts CSV</button>
    <span class="stats" id="stats"></span>
    <div class="help"><kbd>j</kbd>/<kbd>k</kbd> next/prev &nbsp; <kbd>p</kbd> play/pause &nbsp; <kbd>r</kbd> replay
      &nbsp; <kbd>1</kbd> GT correct &nbsp; <kbd>2</kbd> GT wrong &nbsp; <kbd>3</kbd> unsure &nbsp;
      Red = in ground truth but not predicted; green = predicted but not in ground truth.</div>
  </div>
</header>
<main id="list"></main>
<script>
const DATA = __DATA__;
const VERDICTS = [["gt_ok", "GT correct"], ["gt_wrong", "GT wrong"], ["unsure", "Unsure"]];
let reviews = {};
try { reviews = JSON.parse(localStorage.getItem(DATA.storageKey) || "{}"); } catch (e) {}
function save() { try { localStorage.setItem(DATA.storageKey, JSON.stringify(reviews)); } catch (e) {} }

const $ = (s, el = document) => el.querySelector(s);
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const fmt = v => v == null ? "–" : Number.isInteger(v) ? v : v.toFixed(3);
const plain = toks => toks.map(t => t[0]).join(" ");
const tokens = (toks, cls) => `<div class="text ${cls}">${toks.length ? toks.map(([w, c]) =>
  c ? `<span class="chg">${esc(w)}</span>` : esc(w)).join(" ") : "<em>(empty)</em>"}</div>`;

DATA.items.forEach((it, i) => { it.order = i; it.haystack = (it.file + " " + plain(it.gt) + " " + plain(it.pred)).toLowerCase(); });
const sortSel = $("#sort");
sortSel.innerHTML = DATA.sortKeys.map(k => `<option value="${k}">${k} ↓</option>`).join("");

function card(it, rank) {
  const r = reviews[it.file] || {};
  const m = it.metrics;
  const metricRows = DATA.metricInfo.filter(([k]) => k in m).map(([k, d]) =>
    `<tr title="${esc(d)}"><td>${k}</td><td class="num ${k === "confident_errors" && it.flagged ? "hi" : ""}">${fmt(m[k])}</td></tr>`).join("");
  const modelRows = it.models.map(md => {
    const w = md.wer == null ? 0 : Math.min(md.wer, 1);
    return `<tr><td>${esc(md.name)}</td><td style="width:40%"><div class="wbar"><span class="${w >= 0.25 ? "hi" : ""}" style="width:${w * 100}%"></span></div></td><td class="num">${fmt(md.wer)}</td></tr>`;
  }).join("");
  const modelTexts = it.models.filter(md => md.tokens).map(md =>
    `<div class="label">${esc(md.name)} (WER ${fmt(md.wer)})</div>${tokens(md.tokens, "")}`).join("");
  return `<section class="card ${r.verdict ? "v-" + r.verdict : ""}" data-file="${esc(it.file)}">
  <div class="top">
    <span class="rank">#${rank}</span><span class="fname">${esc(it.file)}</span>
    ${it.flagged ? '<span class="badge flag">flagged</span>' : '<span class="badge ok">not flagged</span>'}
    ${it.audio ? `<audio controls preload="none" src="${esc(it.audio)}"></audio>` : '<span class="noaudio">audio not found</span>'}
  </div>
  <div class="grid">
    <div>
      <div class="label">Ground truth</div>${tokens(it.gt, "gt")}
      <div class="label">Predicted (ROVER of ${it.num_models ?? it.models.length} models)</div>${tokens(it.pred, "")}
      ${it.suspect.length ? `<div class="label">Suspect words</div><div class="suspect">${it.suspect.map(s =>
        `<span class="chip ${s[0] === "+" ? "plus" : "minus"}" title="${s[0] === "+" ? "in annotation, no model heard it" : "every model heard it, missing from annotation"}">${esc(s)}</span>`).join("")}</div>` : ""}
      ${modelTexts ? `<details class="models"><summary>Each model's transcript</summary>${modelTexts}</details>` : ""}
    </div>
    <div>
      <div class="label">Scores</div><table>${metricRows}</table>
      <div class="label">WER per model</div><table>${modelRows}</table>
    </div>
  </div>
  <div class="review">
    ${VERDICTS.map(([v, l]) => `<button data-v="${v}" class="${v} ${r.verdict === v ? "on" : ""}">${l}</button>`).join("")}
    <button data-copy title="Copy the predicted transcript into the correction box">Use predicted ↓</button>
    <textarea placeholder="Corrected transcript / notes (optional)">${esc(r.correction || "")}</textarea>
  </div>
</section>`;
}

let visible = [], current = -1;
function render() {
  const key = sortSel.value, filter = $("#filter").value, q = $("#search").value.trim().toLowerCase();
  visible = DATA.items.filter(it => {
    const v = (reviews[it.file] || {}).verdict;
    if (filter === "flagged" && !it.flagged) return false;
    if (filter === "unreviewed" && v) return false;
    if ((filter === "gt_wrong" || filter === "unsure") && v !== filter) return false;
    return !q || it.haystack.includes(q);
  }).sort((a, b) => ((b.metrics[key] ?? -1) - (a.metrics[key] ?? -1)) ||
    ((b.metrics.rover_wer ?? -1) - (a.metrics.rover_wer ?? -1)) || a.order - b.order);
  $("#list").innerHTML = visible.length ? visible.map((it, i) => card(it, i + 1)).join("")
    : '<div class="empty">No utterances match.</div>';
  current = -1;
  stats();
}
function stats() {
  const n = DATA.items.length, done = DATA.items.filter(it => (reviews[it.file] || {}).verdict).length;
  const wrong = DATA.items.filter(it => (reviews[it.file] || {}).verdict === "gt_wrong").length;
  $("#stats").textContent = `${visible.length} shown · ${done}/${n} reviewed · ${wrong} marked GT wrong`;
}
function setReview(file, patch) {
  reviews[file] = Object.assign(reviews[file] || {}, patch);
  save(); stats();
}
function setVerdict(cardEl, v) {
  const file = cardEl.dataset.file, r = reviews[file] || {};
  const nv = r.verdict === v ? null : v;
  setReview(file, {verdict: nv});
  cardEl.className = "card" + (nv ? " v-" + nv : "") + (cardEl.classList.contains("current") ? " current" : "");
  cardEl.querySelectorAll("button[data-v]").forEach(b => b.classList.toggle("on", b.dataset.v === nv));
}

$("#list").addEventListener("click", e => {
  const cardEl = e.target.closest(".card"); if (!cardEl) return;
  focusCard([...document.querySelectorAll(".card")].indexOf(cardEl), false);
  const b = e.target.closest("button"); if (!b) return;
  if (b.dataset.v) setVerdict(cardEl, b.dataset.v);
  if ("copy" in b.dataset) {
    const it = DATA.items.find(x => x.file === cardEl.dataset.file), ta = $("textarea", cardEl);
    ta.value = plain(it.pred); setReview(it.file, {correction: ta.value});
  }
});
$("#list").addEventListener("input", e => {
  if (e.target.tagName === "TEXTAREA") setReview(e.target.closest(".card").dataset.file, {correction: e.target.value});
});
// Only one clip plays at a time.
document.addEventListener("play", e => document.querySelectorAll("audio").forEach(a => a !== e.target && a.pause()), true);

function focusCard(i, scroll = true) {
  const cards = document.querySelectorAll(".card");
  if (!cards.length) return;
  i = Math.max(0, Math.min(cards.length - 1, i));
  if (cards[current]) cards[current].classList.remove("current");
  current = i; cards[i].classList.add("current");
  if (scroll) cards[i].scrollIntoView({behavior: "smooth", block: "start"});
}
document.addEventListener("keydown", e => {
  if (e.target.matches("input, textarea, select") || e.ctrlKey || e.metaKey || e.altKey) return;
  const cards = document.querySelectorAll(".card"), c = cards[current], audio = c && $("audio", c);
  if (e.key === "j") { document.querySelectorAll("audio").forEach(a => a.pause()); focusCard(current + 1); }
  else if (e.key === "k") { document.querySelectorAll("audio").forEach(a => a.pause()); focusCard(current - 1); }
  else if (e.key === "p" && audio) { audio.paused ? audio.play() : audio.pause(); }
  else if (e.key === "r" && audio) { audio.currentTime = 0; audio.play(); }
  else if ("123".includes(e.key) && c) { setVerdict(c, VERDICTS[+e.key - 1][0]); }
  else return;
  e.preventDefault();
});

$("#export").addEventListener("click", () => {
  const q = s => `"${String(s ?? "").replace(/"/g, '""')}"`;
  const lines = [["file_name", "verdict", "ground_truth", "predicted", "correction", "confident_errors", "flagged"].join(",")];
  DATA.items.forEach(it => {
    const r = reviews[it.file] || {};
    lines.push([it.file, r.verdict || "", plain(it.gt), plain(it.pred), r.correction || "",
      it.metrics.confident_errors ?? "", it.flagged].map(q).join(","));
  });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lines.join("\n") + "\n"], {type: "text/csv"}));
  a.download = "label_review_verdicts.csv"; a.click();
});
[sortSel, $("#filter")].forEach(el => el.addEventListener("change", render));
$("#search").addEventListener("input", render);
document.title = "Label Error Review – " + DATA.title;
render();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(
        description="Build an HTML page to review suspected annotation errors.")
    parser.add_argument("--scores-csv", default="label_error_scores.csv")
    parser.add_argument("--ground-truth-csv", required=True)
    parser.add_argument("--audio-dir", required=True,
                        help="Folder with the audio files (searched recursively).")
    parser.add_argument("--transcripts", nargs="*", default=[],
                        help="Optional per-model transcript CSVs to show each model's output.")
    parser.add_argument("--output-html", default="label_error_review.html")
    parser.add_argument("--embed-audio", action="store_true",
                        help="Embed audio in the HTML (portable, but large).")
    args = parser.parse_args()
    build_review_page(
        scores_csv=args.scores_csv,
        ground_truth_csv=args.ground_truth_csv,
        audio_dir=args.audio_dir,
        output_html=args.output_html,
        transcript_csvs=args.transcripts,
        embed_audio=args.embed_audio,
    )


if __name__ == "__main__":
    main()
