# ASR annotation pipeline

Two jobs, using an ensemble of speech-recognition models:

1. **Audit:** you have transcripts (annotations) and want to find the wrong ones.
2. **Generate:** you have audio but no transcripts, and want to create them with as little human work as possible.

Both start the same way: transcribe the audio with several models, then combine their outputs with **ROVER**, a word-by-word majority vote between the models.

## What we found (LibriSpeech, 250 utterances)

- **Models:** `facebook/wav2vec2-large-960h-lv60-self`, `facebook/hubert-large-ls960-ft`, `openai/whisper-large-v3-turbo`.
- **ROVER accuracy:** 1.44% word error rate (WER) against the true labels, better than any single model (1.62–1.90%).
- **Audit:** with 25 errors planted in 250 annotations, the audit found **24/25** (precision 0.83).
  - Scoring by error *rate* (WER/CER) found only 12/25, because a 2-word error in a long sentence looks tiny as a rate. Counting the words the models *agree* the annotation got wrong works much better.
- **Generate:** when all 3 models agree (70% of utterances), the ROVER transcript has 0.43% WER and needs no human. The other 30% go to a person.

## One-time setup

Run everything in WSL, from `~/aiverify2/asr_testing`, with the `khaslana` environment. That environment has a working CUDA torch; the system `python3` doesn't.

```bash
PY=~/aiverify/khaslana/bin/python
~/aiverify/khaslana/bin/pip install jiwer
```

## Step 1 (both jobs): transcribe with each model

Run this once per model. Each run writes `file_name,annotation` for every audio file in the folder.

```bash
$PY transcribe.py --model facebook/wav2vec2-large-960h-lv60-self \
    --audio-dir <AUDIO_DIR> --output-csv transcripts/wav2vec2_large.csv --device cuda --fp16

$PY transcribe.py --model facebook/hubert-large-ls960-ft \
    --audio-dir <AUDIO_DIR> --output-csv transcripts/hubert_large.csv --device cuda --fp16

$PY transcribe.py --model openai/whisper-large-v3-turbo \
    --audio-dir <AUDIO_DIR> --output-csv transcripts/whisper_turbo.csv --device cuda --fp16 \
    --language en --chunk-length-s 30
```

- **The model name decides where it runs:** any Hugging Face model id or local model folder runs on your GPU. A URL (`http://...`) sends the audio to a model server instead.
- **Use at least 3 models,** so that votes can break ties.

## Job A: annotation audit

```
audio ─► transcribe.py ×3 ─► ROVER vote ─┐
                                         ├─► compare word by word ─► flagged rows
your annotations CSV ────────────────────┘
```

```bash
$PY detect_label_errors.py --ground-truth-csv <YOUR_LABELS.csv> \
    --transcripts transcripts/wav2vec2_large.csv transcripts/hubert_large.csv transcripts/whisper_turbo.csv
```

- **Input:** your labels as `file_name,annotation`.
- **Output:** `label_error_scores.csv`, worst first. Check the rows where `flagged` is True.
- **`suspect_words` shows why a row was flagged:**
  - `+WORD` is in the annotation, but no model heard it.
  - `-WORD` was heard by every model, but is missing from the annotation.
- **Default rule:** flag a row if at least 2 such words are found. Change it with `--threshold`.
- **Expect some false flags:** foreign-language speech, unusual names and compound words ("LOIN CLOTH" vs "LOINCLOTH") get flagged even when the annotation is right. Use your judgement.
- **To measure the audit itself:** `make_planted_errors.py` plants known errors into a correct label set and writes an answer key. Pass `--planted-key planted_errors_key.csv` to see how many get caught.

**Review the results by listening:**
```bash
$PY review_label_errors.py --ground-truth-csv <YOUR_LABELS.csv> --audio-dir <AUDIO_DIR> \
    --transcripts transcripts/wav2vec2_large.csv transcripts/hubert_large.csv transcripts/whisper_turbo.csv
```
- **Output:** `label_error_review.html`, one card per utterance, most suspicious first. Open it in a browser.
- **Each card has:** an audio player, the annotation and the ROVER transcript with differing words highlighted, the `suspect_words`, and the scores (including each model's WER). `--transcripts` is optional and adds each model's own transcript.
- **Mark each annotation** as correct, wrong or unsure, and optionally type a correction. Keys: `j`/`k` next/prev, `p` play/pause, `r` replay, `1`/`2`/`3` verdicts.
- **Save your work with "Export verdicts CSV".** Verdicts are also kept in the browser, but that is lost if site data is cleared.
- **Keep the HTML where it was written:** it links to the audio by relative path. Add `--embed-audio` for one portable (but large) file.

## Job B: annotation generation (no labels yet)

```
audio ─► transcribe.py ×3 ─► ROVER vote ─► how many words do the models disagree on?
                                              0 words   ─► auto_labels.csv   (done)
                                              1–2 words ─► review_queue.csv  (human checks the [A|B] words)
                                              3+ words  ─► review_queue.csv  (human transcribes, using the draft)
                                                               │
                                     auto_labels + filled queue ─► merge ─► final_labels.csv
```

**1. Build:**
```bash
$PY pseudo_label.py build --output-dir pseudo_labels \
    --transcripts transcripts/wav2vec2_large.csv transcripts/hubert_large.csv transcripts/whisper_turbo.csv
```

**2. Have a person fill in the `human_annotation` column of `pseudo_labels/review_queue.csv`:**
- **`review` rows:** listen and check only the bracketed words. `[PLANE|PLAN]` means the models disagreed between PLANE and PLAN; `-` means "no word". The `label` column holds the models' best guess, ready to copy and correct.
- **`transcribe` rows:** listen and write the whole sentence. Use `label` as a starting point.
- **Any capitalisation or punctuation is fine.** It is converted to LibriSpeech style (uppercase, no punctuation) automatically.

**3. Merge:**
```bash
$PY pseudo_label.py merge --output-dir pseudo_labels
```
This writes `pseudo_labels/final_labels.csv`. It stops if any queue row is still empty; add `--allow-incomplete` to write only the finished rows.

**Before trusting this on a new kind of audio,** hand-label about 100–200 of its utterances, then rerun `build` with `--true-labels <hand_labels.csv>`. The report shows how accurate the auto labels are. If they aren't accurate enough, send more to humans with `--auto-max-contested` and `--review-max-contested`.

## Files

| File | What it does |
|---|---|
| `transcribe.py` | Transcribes an audio folder with one model and writes a CSV |
| `detect_label_errors.py` | Job A: ROVER plus scoring of existing annotations |
| `review_label_errors.py` | Job A: HTML page to listen to each utterance and check its annotation, worst first |
| `pseudo_label.py` | Job B: `build` (auto labels + review queue) and `merge` (final labels) |
| `make_planted_errors.py` | Plants known errors into a correct label set to test Job A |
| `ground_truth_transcribe_test.csv`, `planted_errors_key.csv` | The test set: 250 LibriSpeech labels with 25 planted errors, and their answer key |

## Known limits

- **Accuracy depends on the audio:** the numbers above are on clean, read English audio (LibriSpeech), which suits these models. Noisy, accented or domain-specific speech will route more to humans and leave more errors in the auto labels. Calibrate first (see Job B).
- **Errors shared by all models get through:** the auto labels inherit mistakes all three models make, mostly homophones (PLAN/PLANE) and the spelling of names.
- **British vs. American spelling** isn't controlled (PROGRAM/PROGRAMME), and "O" vs "OH" also varies.
- **Numbers are spelled out** in the generated labels. If a digit survives the vote, the row is sent to a human.
