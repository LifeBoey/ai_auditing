"""
Build a test ground-truth CSV with planted annotation errors.

    LibriSpeech_250/labels.csv  (file_name,annotation,speaker_id)
        -> ground_truth_transcribe_test.csv  (file_name,annotation)
           with NUM_ERRORS annotations corrupted
        -> planted_errors_key.csv  (file_name,error_type,original,corrupted)
           the answer key, for scoring how many planted errors get detected

Error types range from blatant to subtle, so detection can be judged per type:

    swap          annotation replaced by another utterance's annotation
    random_words  every word replaced by a random corpus word (same length)
    substitute    2 words replaced by random corpus words
    delete        2 words removed
    insert        2 random corpus words inserted
"""

import csv
import random

SOURCE_CSV = "nsc_noisy.csv"
OUTPUT_CSV = "nscp_noisy.csv"
KEY_CSV = "planted_errors_key_nscp_noisy.csv"
NUM_ERRORS = 25
NUM_EDITS = 2  # words touched by substitute / delete / insert
SEED = 42
ERROR_TYPES = ["swap", "random_words", "substitute", "delete", "insert"]


def corrupt(text, error_type, all_annotations, vocab, rng):
    words = text.split()

    if error_type == "swap":
        return rng.choice([a for a in all_annotations if a != text])

    if error_type == "random_words":
        return " ".join(rng.choice(vocab) for _ in words)

    # Too short to delete NUM_EDITS words and keep at least one: substitute instead.
    if error_type == "delete" and len(words) <= NUM_EDITS:
        error_type = "substitute"

    if error_type == "substitute":
        for i in rng.sample(range(len(words)), min(NUM_EDITS, len(words))):
            words[i] = rng.choice([v for v in vocab if v != words[i]])
    elif error_type == "delete":
        for i in sorted(rng.sample(range(len(words)), NUM_EDITS), reverse=True):
            del words[i]
    elif error_type == "insert":
        for _ in range(NUM_EDITS):
            words.insert(rng.randint(0, len(words)), rng.choice(vocab))
    else:
        raise ValueError(f"Unknown error type {error_type!r}")
    return " ".join(words)


def main():
    rng = random.Random(SEED)

    with open(SOURCE_CSV, newline="", encoding="utf-8") as fh:
        rows = [{"file_name": r["file_name"], "annotation": r["annotation"]}
                for r in csv.DictReader(fh)]

    all_annotations = [r["annotation"] for r in rows]
    vocab = sorted({w for a in all_annotations for w in a.split()})

    # Spread the error types evenly over the sampled rows.
    picked = rng.sample(range(len(rows)), NUM_ERRORS)
    key = []
    for n, idx in enumerate(picked):
        error_type = ERROR_TYPES[n % len(ERROR_TYPES)]
        original = rows[idx]["annotation"]
        corrupted = original
        while corrupted == original:
            corrupted = corrupt(original, error_type, all_annotations, vocab, rng)
        rows[idx]["annotation"] = corrupted
        key.append({"file_name": rows[idx]["file_name"], "error_type": error_type,
                    "original": original, "corrupted": corrupted})

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file_name", "annotation"])
        writer.writeheader()
        writer.writerows(rows)

    key.sort(key=lambda r: r["file_name"])
    with open(KEY_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["file_name", "error_type", "original", "corrupted"]
        )
        writer.writeheader()
        writer.writerows(key)

    print(f"Wrote {len(rows)} rows ({NUM_ERRORS} corrupted) to {OUTPUT_CSV}")
    print(f"Wrote answer key to {KEY_CSV}")


if __name__ == "__main__":
    main()
