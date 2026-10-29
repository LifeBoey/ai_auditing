import json
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
from audiomentations import AddColorNoise, AddGaussianSNR

# Paths relative to this script
BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "nsc_test"
CSV_FILE = BASE_DIR / "nsc_test.csv"

OUTPUT_DIR = BASE_DIR / "nsc_noisy"
OUTPUT_CSV = BASE_DIR / "nsc_noisy.csv"
LOG_FILE = BASE_DIR / "nsc_noisy_log.json"

SNR_DB = 10.0
RANDOM_SEED = 42  # Reproducible noise; change or set to None for different output


def color_noise(f_decay):
    """Colored noise at a fixed SNR and fixed spectral decay."""
    return AddColorNoise(
        min_snr_db=SNR_DB,
        max_snr_db=SNR_DB,
        min_f_decay=f_decay,
        max_f_decay=f_decay,
        p=1.0,
    )


# One transform is picked uniformly at random per file
NOISE_POOL = {
    "white": AddGaussianSNR(min_snr_db=SNR_DB, max_snr_db=SNR_DB, p=1.0),
    "pink": color_noise(-3.01),
    "brown": color_noise(-6.02),
    "blue": color_noise(3.01),
    "violet": color_noise(6.02),
}


def to_jsonable(params):
    """Convert numpy scalars in transform parameters to plain Python types."""
    return {k: v.item() if isinstance(v, np.generic) else v for k, v in params.items()}


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    OUTPUT_DIR.mkdir(exist_ok=True)

    wav_files = sorted(p for p in AUDIO_DIR.iterdir() if p.suffix.lower() == ".wav")
    log = {"snr_db": SNR_DB, "seed": RANDOM_SEED, "files": {}}

    for wav_path in wav_files:
        samples, sample_rate = sf.read(wav_path, dtype="float32")
        info = sf.info(wav_path)

        noise_name = random.choice(list(NOISE_POOL))
        transform = NOISE_POOL[noise_name]
        noisy = transform(samples=samples, sample_rate=sample_rate)
        noisy = np.clip(noisy, -1.0, 1.0)

        sf.write(OUTPUT_DIR / wav_path.name, noisy, sample_rate, subtype=info.subtype)

        log["files"][wav_path.name] = {
            "noise": noise_name,
            "params": to_jsonable(transform.parameters),
        }

    # Noise does not change the spoken words, so annotations carry over unchanged
    shutil.copyfile(CSV_FILE, OUTPUT_CSV)

    counts = Counter(entry["noise"] for entry in log["files"].values())
    log["summary"] = dict(sorted(counts.items()))

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    print(f"Processed {len(wav_files)} files -> {OUTPUT_DIR}")
    print(f"Noise counts: {log['summary']}")
    print(f"Annotations copied to {OUTPUT_CSV}")
    print(f"Log written to {LOG_FILE}")


if __name__ == "__main__":
    main()
