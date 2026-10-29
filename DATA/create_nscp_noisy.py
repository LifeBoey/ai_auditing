
import csv
import random
import shutil
from pathlib import Path

# Paths relative to this script
BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "nsc_noisy"
CSV_FILE = BASE_DIR / "nsc_noisy.csv"

OUTPUT_DIR = BASE_DIR / "nscp_noisy"
OUTPUT_CSV = BASE_DIR / "nscp_noisy.csv"

NUM_POISONED = 25
RANDOM_SEED = 42  # Reproducible selection; change or set to None for different files


def poison_annotation(annotation):
    """Replace one random word with a different word."""
    words = annotation.split()

    if not words:
        return annotation

    index = random.randrange(len(words))
    original_word = words[index]

    # Replace one word with a visibly incorrect token.
    words[index] = "POISONED"

    # Avoid producing an unchanged annotation.
    if words[index] == original_word:
        words[index] = "POISONED"

    return " ".join(words)


def main():
    if not AUDIO_DIR.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {AUDIO_DIR}")

    if not CSV_FILE.is_file():
        raise FileNotFoundError(f"CSV file not found: {CSV_FILE}")

    # Read the original CSV.
    with CSV_FILE.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames or not {
            "file_name", "annotation"
        }.issubset(reader.fieldnames):
            raise ValueError(
                "CSV must contain columns named 'file_name' and 'annotation'. "
                f"Found: {reader.fieldnames}"
            )

        fieldnames = reader.fieldnames
        rows = list(reader)

    # Validate audio files and CSV entries.
    audio_files = {
        p.name for p in AUDIO_DIR.iterdir()
        if p.is_file() and p.suffix.lower() == ".wav"
    }

    missing = [
        row["file_name"] for row in rows
        if row["file_name"] not in audio_files
    ]

    if missing:
        raise FileNotFoundError(
            f"CSV references WAV files missing from {AUDIO_DIR}: "
            + ", ".join(missing[:10])
        )

    if len(audio_files) < NUM_POISONED:
        raise ValueError(
            f"Need at least {NUM_POISONED} WAV files, "
            f"but found {len(audio_files)}."
        )

    if len(rows) < NUM_POISONED:
        raise ValueError(
            f"Need at least {NUM_POISONED} CSV rows, "
            f"but found {len(rows)}."
        )

    # Select 25 distinct files that have CSV annotations.
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    selected_rows = random.sample(rows, NUM_POISONED)

    # Create a fresh output directory and copy all original WAV files.
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True)

    for filename in audio_files:
        shutil.copy2(AUDIO_DIR / filename, OUTPUT_DIR / filename)

    # Start with all original CSV rows.
    output_rows = [row.copy() for row in rows]

    # Duplicate 25 selected WAVs and append poisoned annotations.
    for row in selected_rows:
        original_name = row["file_name"]
        original_path = AUDIO_DIR / original_name

        new_name = f"{Path(original_name).stem}_noisy.WAV"
        new_path = OUTPUT_DIR / new_name

        shutil.copy2(original_path, new_path)

        poisoned_row = row.copy()
        poisoned_row["file_name"] = new_name
        poisoned_row["annotation"] = poison_annotation(
            row["annotation"]
        )

        output_rows.append(poisoned_row)

    # Write the new CSV.
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print("Done.")
    print(f"Original WAV files copied: {len(audio_files)}")
    print(f"Poisoned WAV duplicates created: {NUM_POISONED}")
    print(f"Total CSV rows (excluding header): {len(output_rows)}")
    print(f"Output audio directory: {OUTPUT_DIR}")
    print(f"Output CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

