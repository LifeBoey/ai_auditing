"""
Transcribe a directory of audio files with one ASR model and write a CSV:

    file_name,annotation
    5338-284437-0033.flac,I'LL LOOK IN THE GREAT BOOK FIRST

Phase 1 of a multi-model ROVER ensemble: call transcribe_audio_files() once per
model, each writing its own CSV. Every run scans the same directory in the same
sorted order, so the CSVs line up row-for-row (and can also be joined on
file_name, which matches the ground-truth CSVs like LibriSpeech_by_speaker.csv).

Models run locally by default (HF Hub id or local model dir). Sending audio to
a model API is opt-in: only a model string starting with http:// or https:// is
treated as an API.

Example (local, one model after another):

    from transcribe import transcribe_audio_files

    for name, model in [
        ("wav2vec2", "facebook/wav2vec2-base-960h"),
        ("whisper",  "openai/whisper-small"),
    ]:
        transcribe_audio_files(model, "LibriSpeech_by_speaker_flat",
                               f"transcripts/{name}.csv")

Text is written raw (no normalization); normalize at scoring/ROVER time.
"""

import gc
import io
import os
import sys
import csv
import argparse
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

TARGET_SR = 16000
AUDIO_EXTS = (".flac", ".wav", ".mp3", ".ogg", ".m4a")


def find_audio_files(audio_dir, extensions=AUDIO_EXTS, recursive=True):
    """Sorted list of audio paths under audio_dir (sorted for stable row order)."""
    root = Path(audio_dir)
    pattern = "**/*" if recursive else "*"
    exts = {e.lower() for e in extensions}
    files = sorted(
        p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in exts
    )
    if not files:
        raise RuntimeError(f"No audio files {sorted(exts)} found under {audio_dir}")

    # file_name is the bare basename, so it must be unique across subfolders.
    seen = {}
    for p in files:
        if p.name in seen:
            raise RuntimeError(
                f"Duplicate file name {p.name!r}: {seen[p.name]} and {p}"
            )
        seen[p.name] = p
    return files


def load_audio(path, target_sr=TARGET_SR):
    """Read an audio file as a mono float32 waveform at target_sr."""
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio


# ---------------------------------------------------------------------------
# Model -> transcriber: a callable (list of waveforms, sampling_rate) -> texts.
# ---------------------------------------------------------------------------

def _api_transcriber(base_url, timeout=300):
    """POST batches to an app.py-style API (/predict taking an .npz batch)."""
    base_url = base_url.rstrip("/")
    resp = requests.get(f"{base_url}/health", timeout=10)
    resp.raise_for_status()

    def transcribe(audio_list, sampling_rate):
        buffer = io.BytesIO()
        arrays = {f"audio_{i}": np.asarray(a, dtype=np.float32)
                  for i, a in enumerate(audio_list)}
        np.savez(buffer, sampling_rate=np.int64(sampling_rate), **arrays)
        buffer.seek(0)
        resp = requests.post(
            f"{base_url}/predict",
            files={"file": ("batch.npz", buffer, "application/octet-stream")},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["predictions"]

    return transcribe


def _hf_transcriber(model, device=None, batch_size=8, pipeline_kwargs=None,
                    generate_kwargs=None, fp16=False):
    """Hugging Face ASR pipeline: works for wav2vec2, HuBERT, Whisper, etc."""
    import torch
    from transformers import pipeline

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"device={device!r} requested but torch cannot see a GPU "
            f"(torch {torch.__version__}). Check the Python environment."
        )
    pipeline_kwargs = dict(pipeline_kwargs or {})
    if fp16:
        pipeline_kwargs.setdefault("dtype", torch.float16)
    print(f"Loading {model} on {device}"
          + (f" ({torch.cuda.get_device_name(device)})" if str(device).startswith("cuda") else ""))
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        device=device,
        **pipeline_kwargs,
    )

    def transcribe(audio_list, sampling_rate):
        inputs = [{"raw": a, "sampling_rate": sampling_rate} for a in audio_list]
        call_kwargs = {"batch_size": batch_size}
        if generate_kwargs:
            call_kwargs["generate_kwargs"] = generate_kwargs
        return [out["text"] for out in pipe(inputs, **call_kwargs)]

    return transcribe


def _free_memory():
    """Drop unreferenced models and return cached GPU memory, if torch is loaded."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_transcriber(model, device=None, batch_size=8, pipeline_kwargs=None,
                      generate_kwargs=None, api_timeout=300, fp16=False):
    """
    Resolve the `model` argument into a transcriber callable:

        "http://..." / "https://..." -> model API (app.py contract)
        any other string / Path      -> HF Hub id or local model directory
        callable                     -> used as-is: fn(audio_list, sr) -> texts
    """
    if callable(model):
        return model
    model = str(model)
    if model.startswith(("http://", "https://")):
        return _api_transcriber(model, timeout=api_timeout)
    return _hf_transcriber(model, device=device, batch_size=batch_size,
                           pipeline_kwargs=pipeline_kwargs,
                           generate_kwargs=generate_kwargs, fp16=fp16)


def transcribe_audio_files(model, audio_dir, output_csv, batch_size=8,
                           device=None, recursive=True, extensions=AUDIO_EXTS,
                           target_sr=TARGET_SR, pipeline_kwargs=None,
                           generate_kwargs=None, api_timeout=300, fp16=False):
    """
    Transcribe every audio file under audio_dir with one model and write
    output_csv with columns file_name,annotation.

    Args:
        model: HF Hub id or local model dir (e.g. "facebook/wav2vec2-base-960h"),
            an API base URL (e.g. "http://localhost:6969"), or a callable
            fn(list_of_waveforms, sampling_rate) -> list_of_texts.
        audio_dir: Directory containing the audio files.
        output_csv: Path of the CSV to write (parent dirs are created).
        batch_size: Clips per model call.
        device: "cuda"/"cpu" for HF models; auto-detected when None.
        recursive: Also search subdirectories (LibriSpeech nests speaker/chapter).
        extensions: Audio file extensions to include.
        target_sr: Sampling rate audio is resampled to before transcription.
        pipeline_kwargs: Extra kwargs for transformers.pipeline(...).
        generate_kwargs: Generation kwargs for seq2seq models, e.g.
            {"language": "en", "task": "transcribe"} for Whisper.
        api_timeout: Per-request timeout in seconds for API models.
        fp16: Load HF models in float16 (GPU only; ~half the memory, faster).

    Returns:
        list[dict]: rows of {"file_name", "annotation"} in written order.
    """
    files = find_audio_files(audio_dir, extensions, recursive)
    transcribe = build_transcriber(model, device=device, batch_size=batch_size,
                                   pipeline_kwargs=pipeline_kwargs,
                                   generate_kwargs=generate_kwargs,
                                   api_timeout=api_timeout, fp16=fp16)

    model_label = model if isinstance(model, str) else getattr(model, "__name__", "callable")
    print(f"Transcribing {len(files)} files from {audio_dir} with {model_label}")

    rows = []
    for start in range(0, len(files), batch_size):
        batch_files = files[start:start + batch_size]
        audio = [load_audio(p, target_sr) for p in batch_files]
        texts = transcribe(audio, target_sr)
        if len(texts) != len(batch_files):
            raise RuntimeError(
                f"Model returned {len(texts)} transcripts for {len(batch_files)} clips."
            )
        rows.extend(
            {"file_name": p.name, "annotation": t.strip()}
            for p, t in zip(batch_files, texts)
        )
        print(f"  {len(rows)}/{len(files)}")

    # Release the model so the next call (next model) starts with free memory.
    del transcribe
    _free_memory()

    os.makedirs(os.path.dirname(str(output_csv)) or ".", exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file_name", "annotation"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_csv}")
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe a directory of audio files to file_name,annotation CSV."
    )
    parser.add_argument("--model", required=True,
                        help="HF model id, local model dir, or API base URL.")
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None,
                        help="e.g. cuda / cuda:0 / cpu (default: cuda if available).")
    parser.add_argument("--fp16", action="store_true",
                        help="Load the model in float16 (GPU only).")
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--language", default=None,
                        help="Whisper-style models: force the language, e.g. 'en'.")
    parser.add_argument("--chunk-length-s", type=float, default=None,
                        help="Split audio longer than this many seconds (Whisper: 30).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    transcribe_audio_files(
        model=args.model,
        audio_dir=args.audio_dir,
        output_csv=args.output_csv,
        batch_size=args.batch_size,
        device=args.device,
        recursive=not args.no_recursive,
        fp16=args.fp16,
        pipeline_kwargs=(
            {"chunk_length_s": args.chunk_length_s} if args.chunk_length_s else None
        ),
        generate_kwargs=(
            {"language": args.language, "task": "transcribe"} if args.language else None
        ),
    )
