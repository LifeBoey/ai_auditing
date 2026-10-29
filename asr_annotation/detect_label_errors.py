"""
Phase 2: flag possibly-wrong annotations by ensembling several ASR models.

Inputs:
    ground-truth CSV       file_name,annotation   (the annotations under audit)
    transcript CSVs        file_name,annotation   (one per model, from transcribe.py)

For every utterance the model transcripts are combined with ROVER (majority vote
over aligned words), and the annotation is scored against the ensemble:

    confident_errors        (default) max(unsupported_words, missing_words):
      unsupported_words       annotation words that NO model produced
                              (catches insertions and substitutions)
      missing_words           words EVERY model produced that the annotation
                              changed, or where the annotation has no word but
                              MOST models heard one, even if they disagree on
                              which (catches deletions and substitutions)
                            A count, not a rate, so a 2-word error in a long
                            sentence is not diluted, and a word that only some
                            models stumble on is not blamed on the annotation.
    rover_wer / rover_cer   WER/CER of the annotation vs the ROVER transcript
    weighted_wer            like rover_wer, but each disagreement is weighted by
                            how strongly the models agreed on that word (a word
                            all models agree on counts 1.0; a 3-of-5 vote counts
                            0.6), so hard audio where the models themselves
                            disagree is not blamed on the annotation
    min_model_wer           WER vs the single closest model: if the annotation
                            is right, at least one good model usually matches it
    median_model_wer        WER vs the typical model
    model_disagreement      mean WER of each model vs the ROVER transcript (how
                            hard the audio is; context for the scores above)

Rows are written worst-first by the chosen metric, with flagged = metric >= threshold
(default 2 for confident_errors, 0.25 for the rate metrics). The suspect_words
column lists the evidence: "+word" for unsupported, "-word" for missing.
If a planted-error key is given (from make_planted_errors.py), every metric is
evaluated against it so the best metric/threshold can be chosen from data.

All text is normalized before comparison, by default with Whisper's
EnglishTextNormalizer (see whisper_normalizer) so that "Mr." / "MISTER" or
"1990" / "NINETEEN NINETY" are not counted as errors.
"""

import os
import re
import csv
import argparse
import statistics
from collections import Counter
from pathlib import Path

from jiwer import cer

METRICS = ["confident_errors", "rover_wer", "rover_cer", "weighted_wer",
           "min_model_wer", "median_model_wer"]
DEFAULT_THRESHOLDS = {"confident_errors": 2}  # counts; every other metric is a rate
DEFAULT_RATE_THRESHOLD = 0.25
# A word the annotation omits counts as missing when more than this fraction of
# models heard some word there (a strict majority), whether or not they agree
# on which; a substituted word needs all models to agree on it.
GAP_MIN_AGREEMENT = 0.5


def default_threshold(metric):
    return DEFAULT_THRESHOLDS.get(metric, DEFAULT_RATE_THRESHOLD)


WHISPER_TOKENIZER = "openai/whisper-large-v3-turbo"


def basic_normalize(text):
    """Lowercase, strip punctuation and collapse whitespace (as in asr_util)."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def whisper_normalizer(tokenizer_id=WHISPER_TOKENIZER):
    """
    Whisper's EnglishTextNormalizer: lowercases, strips punctuation, expands
    titles ("Mr." -> "mister"), standardizes numbers ("nineteen ninety" and
    "1990" -> "1990"), contractions and British/American spellings. Applied to
    both annotation and transcripts, so formatting differences between models
    (Whisper vs LibriSpeech-style CTC output) are not counted as errors.

    The spelling table ships with the Whisper tokenizer, which is downloaded
    (or read from the HF cache) once here.
    """
    from transformers import WhisperTokenizer
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    tokenizer = WhisperTokenizer.from_pretrained(tokenizer_id)
    english = EnglishTextNormalizer(tokenizer.english_spelling_normalizer)
    # The normalizer can leave double spaces; collapse so .split() is clean.
    return lambda text: re.sub(r"\s+", " ", english(text)).strip()


NORMALIZERS = {"whisper": whisper_normalizer, "basic": lambda: basic_normalize}


# ---------------------------------------------------------------------------
# Word alignment and ROVER
# ---------------------------------------------------------------------------

def align(ref, hyp):
    """
    Levenshtein-align two word lists. Returns a list of (ref_word, hyp_word)
    pairs where None marks a gap (deletion: hyp None, insertion: ref None).
    """
    n, m = len(ref), len(hyp)
    dist = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dist[i][0] = i
    for j in range(m + 1):
        dist[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dist[i][j] = min(
                dist[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]),
                dist[i - 1][j] + 1,
                dist[i][j - 1] + 1,
            )

    pairs = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dist[i][j] == dist[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            pairs.append((ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and dist[i][j] == dist[i - 1][j] + 1:
            pairs.append((ref[i - 1], None))
            i -= 1
        else:
            pairs.append((None, hyp[j - 1]))
            j -= 1
    return pairs[::-1]


def word_errors(ref, hyp):
    """Number of word edits (S + D + I) between two word lists."""
    return sum(r != h for r, h in align(ref, hyp))


def wer_words(ref, hyp):
    return word_errors(ref, hyp) / max(len(ref), 1)


def rover_slots(hyps):
    """
    Align word-list hypotheses into ROVER voting slots.

    The pivot is the medoid hypothesis (fewest total edits to the others). Every
    hypothesis is aligned to the pivot, giving one slot per pivot word plus
    insertion slots in the gaps between them.

    Returns a list of slots; each slot is a list with one vote per hypothesis
    (a word, or None for "no word"), the pivot's vote first.
    """
    if len(hyps) == 1:
        return [[w] for w in hyps[0]]

    pivot_idx = min(
        range(len(hyps)),
        key=lambda k: sum(word_errors(hyps[k], h) for h in hyps),
    )
    pivot = hyps[pivot_idx]
    ordered = [pivot] + [h for k, h in enumerate(hyps) if k != pivot_idx]

    # Per hypothesis: word at each pivot position, and words inserted before it
    # (gap index len(pivot) holds trailing insertions).
    subs, ins = [], []
    for h in ordered:
        h_sub = [None] * len(pivot)
        h_ins = [[] for _ in range(len(pivot) + 1)]
        p = 0
        for ref_w, hyp_w in align(pivot, h):
            if ref_w is None:
                h_ins[p].append(hyp_w)
            else:
                h_sub[p] = hyp_w
                p += 1
        subs.append(h_sub)
        ins.append(h_ins)

    slots = []
    for gap in range(len(pivot) + 1):
        width = max(len(h_ins[gap]) for h_ins in ins)
        for k in range(width):
            slots.append([h_ins[gap][k] if k < len(h_ins[gap]) else None for h_ins in ins])
        if gap < len(pivot):
            slots.append([h_sub[gap] for h_sub in subs])
    return slots


def vote(votes):
    """Majority vote over one slot -> (winner, fraction). Ties go to the first vote."""
    # Counter keeps first-seen order on ties, and the pivot votes first.
    winner, count = Counter(votes).most_common(1)[0]
    return winner, count / len(votes)


def rover(hyps):
    """
    Combine word-list hypotheses by ROVER-style voting (see rover_slots). Each
    slot is decided by majority vote, with "no word" as a valid candidate; ties
    go to the pivot.

    Returns (words, confidences): the voted transcript and, per output word, the
    fraction of hypotheses that voted for it.
    """
    words, confidences = [], []
    for votes in rover_slots(hyps):
        winner, fraction = vote(votes)
        if winner is not None:
            words.append(winner)
            confidences.append(fraction)
    return words, confidences


def word_presence(hyps):
    """
    Per ROVER output word (as returned by rover), the fraction of hypotheses
    that put any word in its slot. Unlike the vote fraction, this is high when
    every model heard something but they disagree on what it was.
    """
    return [sum(v is not None for v in votes) / len(votes)
            for votes in rover_slots(hyps) if vote(votes)[0] is not None]


def weighted_word_errors(annotation, rover_words, confidences):
    """
    Edits between annotation and ROVER output, each weighted by the ensemble's
    confidence in the ROVER word involved. An extra annotation word (no ROVER
    word to weigh it by) counts 1.0.
    """
    total = 0.0
    r = 0
    for ann_w, rov_w in align(annotation, rover_words):
        if rov_w is None:
            total += 1.0
            continue
        if ann_w != rov_w:
            total += confidences[r]
        r += 1
    return total


def split_compounds(pairs):
    """
    Indices of (annotation, ROVER) alignment pairs where the models split one
    annotation word into several ("classrooms" vs "class rooms", "app" vs
    "a p p"): a run of gaps plus the neighbouring aligned word that, joined,
    spell that annotation word. These are spelling differences, not errors.
    """
    split = set()
    i = 0
    while i < len(pairs):
        if pairs[i][0] is not None:
            i += 1
            continue
        lo = hi = i
        while hi + 1 < len(pairs) and pairs[hi + 1][0] is None:
            hi += 1
        gap = "".join(h for _, h in pairs[lo:hi + 1])
        if lo > 0 and pairs[lo - 1][1] is not None and pairs[lo - 1][1] + gap == pairs[lo - 1][0]:
            split.update(range(lo - 1, hi + 1))
        if hi + 1 < len(pairs) and pairs[hi + 1][1] is not None and gap + pairs[hi + 1][1] == pairs[hi + 1][0]:
            split.update(range(lo, hi + 2))
        i = hi + 1
    return split


def confident_disagreements(annotation, hyps, rover_words, confidences, presence):
    """
    Word-level disagreements between the annotation and the models that the
    models are confident about.

    Returns (unsupported, missing):
        unsupported: annotation words that no hypothesis matches when aligned
            to it, i.e. no model heard that word there.
        missing: ROVER words that the annotation replaces with a unanimous
            vote, or omits outright where most models heard some word
            (presence, see word_presence), even if they disagree on which. A
            gap in the annotation where the models heard speech is strong
            evidence of a deletion; a different word there needs every model
            to agree, since models often disagree on spelling (e.g.
            "cuddling" vs "cadlin").
            Words split from an annotation compound are ignored (see
            split_compounds).
    """
    supported = [False] * len(annotation)
    for h in hyps:
        a = 0
        for ann_w, hyp_w in align(annotation, h):
            if ann_w is None:
                continue
            if ann_w == hyp_w:
                supported[a] = True
            a += 1
    unsupported = [w for w, ok in zip(annotation, supported) if not ok]

    missing = []
    r = 0
    pairs = align(annotation, rover_words)
    split = split_compounds(pairs)
    for idx, (ann_w, rov_w) in enumerate(pairs):
        if rov_w is None:
            continue
        if idx in split:
            confident = False
        elif ann_w is None:
            confident = presence[r] > GAP_MIN_AGREEMENT
        else:
            confident = ann_w != rov_w and confidences[r] == 1.0
        if confident:
            missing.append(rov_w)
        r += 1
    return unsupported, missing


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_utterance(annotation, model_texts):
    """
    Score one annotation against the model transcripts (all normalized strings).

    Returns a dict with the ROVER transcript and every metric in METRICS plus
    model_disagreement and per-model WERs ("wer_<model>").
    """
    ann_words = annotation.split()
    hyps = {name: text.split() for name, text in model_texts.items()}

    rover_words, confidences = rover(list(hyps.values()))
    rover_text = " ".join(rover_words)

    model_wers = {name: wer_words(ann_words, h) for name, h in hyps.items()}
    disagreement = statistics.mean(
        wer_words(rover_words, h) if rover_words else float(bool(h))
        for h in hyps.values()
    )
    unsupported, missing = confident_disagreements(
        ann_words, list(hyps.values()), rover_words, confidences,
        word_presence(list(hyps.values()))
    )

    row = {
        "rover_transcript": rover_text,
        "suspect_words": " ".join([f"+{w}" for w in unsupported] + [f"-{w}" for w in missing]),
        "confident_errors": max(len(unsupported), len(missing)),
        "unsupported_words": len(unsupported),
        "missing_words": len(missing),
        "rover_wer": wer_words(ann_words, rover_words),
        "rover_cer": cer(annotation, rover_text) if annotation else 0.0,
        "weighted_wer": weighted_word_errors(ann_words, rover_words, confidences)
                        / max(len(ann_words), 1),
        "min_model_wer": min(model_wers.values()),
        "median_model_wer": statistics.median(model_wers.values()),
        "model_disagreement": disagreement,
    }
    row.update({f"wer_{name}": w for name, w in model_wers.items()})
    return row


def read_annotations(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as fh:
        return {r["file_name"]: r["annotation"] for r in csv.DictReader(fh)}


def detect_label_errors(ground_truth_csv, transcript_csvs, output_csv,
                        metric="confident_errors", threshold=None, normalizer="whisper"):
    """
    Score every annotation in ground_truth_csv against the ensemble of
    transcript_csvs and write output_csv sorted worst-first by `metric`.

    Args:
        ground_truth_csv: CSV (file_name,annotation) of annotations to audit.
        transcript_csvs: Model transcript CSVs (file_name,annotation); each
            model is named after its file stem, e.g. transcripts/whisper.csv
            -> "whisper".
        output_csv: Where the scored, sorted results are written.
        metric: One of METRICS; used for sorting and flagging.
        threshold: flagged = metric >= threshold; None uses the metric's
            default (2 for confident_errors, 0.25 for rates).
        normalizer: Text normalization applied to annotations and transcripts:
            "whisper" (Whisper EnglishTextNormalizer), "basic" (lowercase and
            strip punctuation), or any callable str -> str.

    Returns:
        list[dict]: the scored rows, worst-first.
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
    if threshold is None:
        threshold = default_threshold(metric)
    if isinstance(normalizer, str):
        normalizer = NORMALIZERS[normalizer]()

    ground_truth = read_annotations(ground_truth_csv)
    models = {Path(p).stem: read_annotations(p) for p in transcript_csvs}
    if len(models) != len(transcript_csvs):
        raise ValueError("Transcript CSVs must have distinct file names (model names).")

    rows = []
    skipped = 0
    for file_name, annotation in ground_truth.items():
        model_texts = {
            name: normalizer(texts[file_name])
            for name, texts in models.items() if file_name in texts
        }
        if not model_texts:
            skipped += 1
            continue
        row = {"file_name": file_name, "annotation": annotation,
               "num_models": len(model_texts)}
        row.update(score_utterance(normalizer(annotation), model_texts))
        row["flagged"] = row[metric] >= threshold
        rows.append(row)

    if skipped:
        print(f"WARNING: {skipped} ground-truth files have no transcript from any model.")

    rows.sort(key=lambda r: r[metric], reverse=True)

    fieldnames = (
        ["file_name", "annotation", "rover_transcript", "suspect_words", "flagged",
         "num_models"]
        + METRICS + ["unsupported_words", "missing_words", "model_disagreement"]
        + [f"wer_{name}" for name in models]
    )
    os.makedirs(os.path.dirname(str(output_csv)) or ".", exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)

    n_flagged = sum(r["flagged"] for r in rows)
    print(f"Scored {len(rows)} utterances with {len(models)} models: "
          f"{n_flagged} flagged ({metric} >= {threshold}). Wrote {output_csv}")
    return rows


# ---------------------------------------------------------------------------
# Evaluation against planted errors
# ---------------------------------------------------------------------------

def evaluate_against_key(rows, key_csv, thresholds=None):
    """
    Print how well each metric finds the planted errors in key_csv:

        P@K         precision of the top-K rows, K = number of planted errors
        thr         the metric's threshold (thresholds[metric] or its default)
        prec/recall of flagging metric >= thr
        recall per error type at that threshold
    """
    thresholds = thresholds or {}
    with open(key_csv, newline="", encoding="utf-8") as fh:
        key = {r["file_name"]: r["error_type"] for r in csv.DictReader(fh)}
    planted = set(key) & {r["file_name"] for r in rows}
    k = len(planted)
    error_types = sorted(set(key.values()))

    print(f"\nDetection of {k} planted errors:")
    header = f"{'metric':<18}{'P@K':>7}{'thr':>7}{'flagged':>9}{'prec':>7}{'recall':>8}"
    header += "".join(f"{t[:10]:>12}" for t in error_types)
    print(header)

    for metric in METRICS:
        threshold = thresholds.get(metric, default_threshold(metric))
        ranked = sorted(rows, key=lambda r: r[metric], reverse=True)
        p_at_k = sum(r["file_name"] in planted for r in ranked[:k]) / max(k, 1)

        flagged = {r["file_name"] for r in rows if r[metric] >= threshold}
        tp = len(flagged & planted)
        precision = tp / len(flagged) if flagged else 0.0
        recall = tp / max(k, 1)

        line = (f"{metric:<18}{p_at_k:>7.2f}{threshold:>7g}{len(flagged):>9}"
                f"{precision:>7.2f}{recall:>8.2f}")
        for t in error_types:
            of_type = {f for f in planted if key[f] == t}
            caught = len(of_type & flagged)
            line += f"{f'{caught}/{len(of_type)}':>12}"
        print(line)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Flag possibly-wrong annotations using a ROVER ensemble of ASR transcripts."
    )
    parser.add_argument("--ground-truth-csv", required=True)
    parser.add_argument("--transcripts", nargs="+", required=True,
                        help="One transcript CSV per model (from transcribe.py).")
    parser.add_argument("--output-csv", default="label_error_scores.csv")
    parser.add_argument("--metric", default="confident_errors", choices=METRICS)
    parser.add_argument("--threshold", type=float, default=None,
                        help="flagged = metric >= threshold (default: 2 for "
                             "confident_errors, 0.25 for rate metrics).")
    parser.add_argument("--normalizer", default="whisper", choices=sorted(NORMALIZERS),
                        help="Text normalization before scoring (default: whisper).")
    parser.add_argument("--planted-key", default=None,
                        help="Optional planted_errors_key.csv to evaluate detection.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    scored = detect_label_errors(
        ground_truth_csv=args.ground_truth_csv,
        transcript_csvs=args.transcripts,
        output_csv=args.output_csv,
        metric=args.metric,
        threshold=args.threshold,
        normalizer=args.normalizer,
    )
    if args.planted_key:
        overrides = {args.metric: args.threshold} if args.threshold is not None else None
        evaluate_against_key(scored, args.planted_key, thresholds=overrides)
