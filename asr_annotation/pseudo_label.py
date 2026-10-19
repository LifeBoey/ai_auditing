"""
Generate annotations when none exist: a ROVER ensemble labels the utterances its
models agree on, and the rest go to a human through a CSV queue.

    build   transcripts/*.csv (from transcribe.py) ->
                <output-dir>/auto_labels.csv    file_name,annotation   (no human needed)
                <output-dir>/review_queue.csv   rows for a human to fill in
    merge   auto_labels.csv + filled review_queue.csv ->
                <output-dir>/final_labels.csv   file_name,annotation

Each utterance is put in a tier by its number of contested words (ROVER words
the models did not vote for unanimously):

    auto        0 contested words           ROVER output is the label
    review      1-2 contested words         human checks the [A|B] words only
    transcribe  3+ contested, a model gave  human transcribes, using the
                no text, or a digit is      ROVER draft as a starting point
                left in the output

Agreement is judged on Whisper-normalized text (the "compare view"), so "Mr." vs
"MISTER" or "6" vs "SIX" is not a disagreement. The label text itself comes from
a separate ROVER over LibriSpeech-style text (the "output view": uppercase, no
punctuation, apostrophes kept), where the two CTC models usually outvote
Whisper's formatting.

Calibration: the default cut-offs were chosen on LibriSpeech dev-clean, where
the auto tier came out at ~0.4% WER. For a new kind of data, hand-label
~100-200 utterances and run `build --true-labels` on them first to check that
the auto tier is accurate enough before trusting it.
"""

import os
import re
import csv
import argparse
from pathlib import Path

from detect_label_errors import (
    NORMALIZERS, read_annotations, rover, rover_slots, vote, word_errors,
)

TIERS = ["auto", "review", "transcribe"]


def librispeech_style(text):
    """Uppercase, hyphens/dashes to spaces, keep apostrophes, drop other punctuation."""
    text = text.upper().replace("’", "'").replace("‘", "'")
    text = re.sub(r"[-–—]", " ", text)
    text = re.sub(r"[^A-Z0-9' ]", "", text)
    words = [w.strip("'") for w in text.split()]
    return " ".join(w for w in words if w)


def draft_with_markup(slots):
    """
    Voted text plus a markup version where contested slots read [WINNER|OTHER],
    with "-" standing for "no word", e.g. "BETTER [PLANE|PLAN] THE".
    """
    words, marked = [], []
    for votes in slots:
        winner, fraction = vote(votes)
        if winner is not None:
            words.append(winner)
        if fraction < 1.0:
            options = []
            for v in [winner] + votes:
                if v not in options:
                    options.append(v)
            marked.append("[" + "|".join("-" if v is None else v for v in options) + "]")
        elif winner is not None:
            marked.append(winner)
    return " ".join(words), " ".join(marked)


def needs_transcription(label, out_hyps):
    """A model gave no text, or a digit survived (LibriSpeech spells numbers out)."""
    return any(not h for h in out_hyps) or bool(re.search(r"\d", label))


def assign_tier(contested, forced, auto_max, review_max):
    if forced:
        return "transcribe"
    if contested <= auto_max:
        return "auto"
    if contested <= review_max:
        return "review"
    return "transcribe"


def label_utterances(models, compare, auto_max=0, review_max=2):
    """
    Label every file transcribed by all models.

    Args:
        models: {model_name: {file_name: raw transcript}}.
        compare: normalizer for the compare view (str -> str).

    Returns:
        list[dict] with file_name, tier, num_contested, label (plain ROVER
        text), draft (with [A|B] markup) and raw transcripts per model.
    """
    common = set.intersection(*(set(t) for t in models.values()))
    partial = set.union(*(set(t) for t in models.values())) - common
    if partial:
        print(f"WARNING: {len(partial)} files lack a transcript from some model; skipped.")

    rows = []
    for f in sorted(common):
        raw = {name: texts[f] for name, texts in models.items()}
        out_hyps = [librispeech_style(t).split() for t in raw.values()]
        cmp_hyps = [compare(t).split() for t in raw.values()]

        label, draft = draft_with_markup(rover_slots(out_hyps))
        _, cmp_conf = rover(cmp_hyps)
        contested = sum(c < 1.0 for c in cmp_conf)
        forced = needs_transcription(label, out_hyps)

        rows.append({
            "file_name": f,
            "tier": assign_tier(contested, forced, auto_max, review_max),
            "num_contested": contested,
            "forced": forced,
            "label": label,
            "draft": draft,
            **{f"model_{name}": t for name, t in raw.items()},
        })
    return rows


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    total_words = sum(len(r["label"].split()) for r in rows) or 1
    print(f"\n{'tier':<12}{'utterances':>12}{'share':>9}{'word share':>12}")
    for tier in TIERS:
        sub = [r for r in rows if r["tier"] == tier]
        words = sum(len(r["label"].split()) for r in sub)
        print(f"{tier:<12}{len(sub):>12}{len(sub) / len(rows):>9.1%}{words / total_words:>12.1%}")


def calibration_report(rows, true_csv, compare, auto_max, review_max):
    """Accuracy of the ROVER label per tier, and coverage vs WER per cut-off."""
    true = read_annotations(true_csv)
    scored = [r for r in rows if r["file_name"] in true]
    if not scored:
        print(f"\nNo files in {true_csv} match the transcripts; skipping calibration.")
        return
    for r in scored:
        ref = compare(true[r["file_name"]]).split()
        r["_errs"] = word_errors(ref, compare(r["label"]).split())
        r["_words"] = len(ref)

    def wer(sub):
        return sum(r["_errs"] for r in sub) / max(sum(r["_words"] for r in sub), 1)

    print(f"\nCalibration against {true_csv} ({len(scored)} utterances), "
          f"WER of the ROVER label:")
    print(f"{'tier':<12}{'utterances':>12}{'WER':>9}{'with errors':>13}")
    for tier in TIERS:
        sub = [r for r in scored if r["tier"] == tier]
        print(f"{tier:<12}{len(sub):>12}{wer(sub):>9.2%}{sum(r['_errs'] > 0 for r in sub):>13}")

    print("\nIf auto accepted up to N contested words:")
    print(f"{'N':<4}{'auto share':>12}{'auto WER':>10}")
    for n in range(4):
        sub = [r for r in scored
               if assign_tier(r["num_contested"], r["forced"], n, max(n, review_max)) == "auto"]
        print(f"{n:<4}{len(sub) / len(scored):>12.1%}{wer(sub):>10.2%}")

    wrong = [r for r in scored if r["tier"] == "auto" and r["_errs"]]
    if wrong:
        print(f"\nAuto labels that differ from {true_csv}:")
        for r in wrong:
            print(f"  {r['file_name']}\n    TRUE : {librispeech_style(true[r['file_name']])}"
                  f"\n    LABEL: {r['label']}")


def build(transcript_csvs, output_dir, true_labels=None, auto_max=0, review_max=2,
          normalizer="whisper"):
    """Write auto_labels.csv and review_queue.csv into output_dir (see module doc)."""
    if review_max < auto_max:
        raise ValueError("review_max must be >= auto_max")
    models = {Path(p).stem: read_annotations(p) for p in transcript_csvs}
    if len(models) != len(transcript_csvs):
        raise ValueError("Transcript CSVs must have distinct file names (model names).")
    if len(models) < 3:
        print("WARNING: with fewer than 3 models, votes cannot break disagreements.")
    compare = NORMALIZERS[normalizer]()

    rows = label_utterances(models, compare, auto_max, review_max)

    auto = [{"file_name": r["file_name"], "annotation": r["label"]}
            for r in rows if r["tier"] == "auto"]
    write_csv(Path(output_dir) / "auto_labels.csv", auto, ["file_name", "annotation"])

    queue = sorted((r for r in rows if r["tier"] != "auto"),
                   key=lambda r: (TIERS.index(r["tier"]), r["num_contested"], r["file_name"]))
    for r in queue:
        r["human_annotation"] = ""
    fields = (["file_name", "tier", "num_contested", "draft", "label"]
              + [f"model_{name}" for name in models] + ["human_annotation"])
    write_csv(Path(output_dir) / "review_queue.csv", queue, fields)

    print(f"Labelled {len(rows)} utterances with {len(models)} models "
          f"(auto <= {auto_max} contested words, review <= {review_max}).")
    print_summary(rows)
    print(f"\nWrote {len(auto)} rows to {Path(output_dir) / 'auto_labels.csv'}")
    print(f"Wrote {len(queue)} rows to {Path(output_dir) / 'review_queue.csv'} "
          f"(fill in human_annotation, then run merge)")

    if true_labels:
        calibration_report(rows, true_labels, compare, auto_max, review_max)
    return rows


def merge(output_dir, allow_incomplete=False):
    """Combine auto labels and the filled review queue into final_labels.csv."""
    out = Path(output_dir)
    labels = read_annotations(out / "auto_labels.csv")
    with open(out / "review_queue.csv", newline="", encoding="utf-8") as fh:
        queue = list(csv.DictReader(fh))

    unfilled = []
    for r in queue:
        text = librispeech_style(r.get("human_annotation") or "")
        if not text:
            unfilled.append(r["file_name"])
            continue
        if r["file_name"] in labels:
            raise ValueError(f"{r['file_name']} is in both auto_labels.csv and review_queue.csv")
        labels[r["file_name"]] = text

    if unfilled and not allow_incomplete:
        raise SystemExit(
            f"{len(unfilled)} review_queue.csv rows have no human_annotation, e.g. "
            f"{', '.join(unfilled[:5])}. Fill them in, or pass --allow-incomplete "
            f"to write only the finished rows."
        )

    rows = [{"file_name": f, "annotation": a} for f, a in sorted(labels.items())]
    write_csv(out / "final_labels.csv", rows, ["file_name", "annotation"])
    print(f"Wrote {len(rows)} rows to {out / 'final_labels.csv'}"
          + (f" ({len(unfilled)} unfilled queue rows left out)" if unfilled else ""))
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate annotations with a ROVER ensemble plus a human review queue."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="Write auto_labels.csv and review_queue.csv.")
    b.add_argument("--transcripts", nargs="+", required=True,
                   help="One transcript CSV per model (from transcribe.py).")
    b.add_argument("--output-dir", default="pseudo_labels")
    b.add_argument("--true-labels", default=None,
                   help="Optional hand-labelled CSV (file_name,annotation) for calibration.")
    b.add_argument("--auto-max-contested", type=int, default=0)
    b.add_argument("--review-max-contested", type=int, default=2)
    b.add_argument("--normalizer", default="whisper", choices=sorted(NORMALIZERS))

    m = sub.add_parser("merge", help="Combine auto labels and the filled queue.")
    m.add_argument("--output-dir", default="pseudo_labels")
    m.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.command == "build":
        build(args.transcripts, args.output_dir, args.true_labels,
              args.auto_max_contested, args.review_max_contested, args.normalizer)
    else:
        merge(args.output_dir, args.allow_incomplete)
