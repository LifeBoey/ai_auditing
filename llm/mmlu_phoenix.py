import os, re
import numpy as np
import pandas as pd
from datasets import load_dataset
from anthropic import Anthropic
import tinyBenchmarks as tb
from phoenix.otel import register
from phoenix.client import Client
from openinference.instrumentation.anthropic import AnthropicInstrumentor

import json
records = {}   # idx -> dict with raw text, parsed letter

def parse_letter(text):
    # prefer an explicit "Answer: X" style, else the last standalone A-D
    m = re.findall(r"(?:answer\s*(?:is|:)\s*\(?)([ABCD])\b", text, flags=re.I)
    if not m:
        m = re.findall(r"\b([ABCD])\b", text)
    return m[-1].upper() if m else "?"

import json, os

OUT = "tinymmlu_records.jsonl"
if os.path.exists(OUT):
    os.remove(OUT)

def task(input):
    idx = int(input["idx"])
    r = claude.messages.create(
        model=MODEL, max_tokens=4000,
        messages=[{"role": "user", "content": input["prompt"]
                   + "\nEnd your reply with 'Answer: X' where X is the letter."}],
    )
    text = "".join(b.text for b in r.content if b.type == "text").strip()
    letter = parse_letter(text)
    with open(OUT, "a") as f:
        f.write(json.dumps({"idx": idx, "raw": text, "pred": letter,
                            "stop": r.stop_reason}) + "\n")
    return letter

BASE  = os.environ["YOUR_BASE_URL_VAR"].rstrip("/")
TOKEN = os.environ["YOUR_AUTH_TOKEN_VAR"]
MODEL = os.environ["YOUR_MODEL_VAR"]

# --- tracing: every Claude call shows up in Phoenix ---
tracer_provider = register(project_name="tinymmlu-claude", batch=True)
AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)
claude = Anthropic(base_url=BASE, auth_token=TOKEN)

# --- data: keep the original order! ---
tiny = load_dataset("tinyBenchmarks/tinyMMLU")["test"]
assert len(tiny) == 100, f"expected 100 rows, got {len(tiny)}"

LETTERS = "ABCD"
rows = []
for i, ex in enumerate(tiny):
    opts = "\n".join(f"{LETTERS[j]}. {c}" for j, c in enumerate(ex["choices"]))
    prompt = (
        f"The following is a multiple choice question about {ex['subject'].replace('_', ' ')}.\n\n"
        f"{ex['question']}\n{opts}\n\n"
        "Answer with only the letter (A, B, C, or D)."
    )
    rows.append({"idx": i, "prompt": prompt, "subject": ex["subject"],
                 "answer": LETTERS[ex["answer"]]})
df = pd.DataFrame(rows)

# --- task: what Phoenix runs on each example ---
preds = {}   # idx -> predicted letter (side channel so we can rebuild y in order)

# def task(input):
#     r = claude.messages.create(
#         model=MODEL, max_tokens=20,
#         messages=[{"role": "user", "content": input["prompt"]}],
#     )

#     text = "".join(b.text for b in r.content if b.type == "text").strip()
#     m = re.search(r"\b([ABCD])\b", text)
#     letter = m.group(1) if m else "?"
#     preds[input["idx"]] = letter
#     return letter

def correct(output, expected):
    return 1.0 if output == expected["answer"] else 0.0

# --- Phoenix dataset + experiment ---
px = Client()
dataset = px.datasets.create_dataset(
    name="tinyMMLU-100",
    dataframe=df,
    input_keys=["idx", "prompt", "subject"],
    output_keys=["answer"],
)
px.experiments.run_experiment(
    dataset=dataset, task=task, evaluators=[correct],
    experiment_name="claude-tinymmlu-zero-shot",
)

# --- tinyBenchmarks estimate ---
y = np.array([1 if preds.get(i) == df.loc[i, "answer"] else 0 for i in range(100)])
print("unparseable answers:", sum(1 for v in preds.values() if v == "?"))
print("raw accuracy on the 100 items:", y.mean())
print(tb.evaluate(y, "mmlu"))


y = np.array([1 if records.get(i, {}).get("pred") == df.loc[i, "answer"] else 0
              for i in range(100)])
print("missing records:", 100 - len(records))
print("unparseable:", sum(1 for r in records.values() if r["pred"] == "?"))
print("stop reasons:", pd.Series([r["stop"] for r in records.values()]).value_counts().to_dict())
print("raw accuracy:", y.mean())
json.dump(records, open("tinymmlu_records.json", "w"), indent=2)

print(len(preds), list(preds.items())[:3])   # old side-channel
print(len(records))                           # new one

recs = {}
for line in open(OUT):
    d = json.loads(line)
    recs[d["idx"]] = d

print("records:", len(recs))
y = np.array([1 if recs.get(i, {}).get("pred") == df.loc[i, "answer"] else 0
              for i in range(100)])
print("unparseable:", sum(r["pred"] == "?" for r in recs.values()))
print("stop reasons:", pd.Series([r["stop"] for r in recs.values()]).value_counts().to_dict())
print("raw accuracy:", y.mean())