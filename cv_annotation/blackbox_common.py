"""
The contract between annotation-audit black boxes and ensemble_audit.py.

A black box is any script that judges how suspicious each image's ANNOTATIONS are and
writes two files into its own folder:

  scores.csv   one row per image
                 file_name   key, as in labels.json
                 file_path   absolute path to the image
                 s_<signal>  one or more scores: HIGHER = MORE SUSPICIOUS, any scale
                             (probabilities, counts, ...); blank = no opinion on that image
                 flagged     True / False: the method's own verdict
                 reason      short text for the worst issue it found (blank if none)
  meta.json    {"name":    unique name, e.g. "cleanlab_rcnn"
                "method":  e.g. "cleanlab"
                "group":   the independent evidence it relies on (e.g. "rcnn" for anything
                           built on the Faster R-CNN predictions); methods sharing a group
                           are correlated and the ensemble counts them as one voice
                "signals": {"s_<signal>": what it measures}
                "weights": optional {"s_<signal>": default weight in the method score}; missing
                           signals weigh 1, 0 ignores a signal (ensemble --weights overrides)
                "params":  the settings it ran with}

The ensemble only reads these files, so adding a method means writing them, nothing else.

How a black box is run (run_audit.py): it's a script, found through blackboxes/<method>/blackbox_config.json
(see blackboxes/README.md), started as its own process from the folder the audit runs in, with
  --labels, --images-dir, --out-dir                   always
  --predictions <one predictions file>                if blackbox_config.json says "predictions": "per_model"
  --work-dir, --evaluate-planted, --audit-name        only those its blackbox_config.json lists in "accepts"
  anything given with run_audit.py --box-args <method>="..."
and this folder on PYTHONPATH, so `from blackbox_common import ...` works from anywhere.
run_audit.py checks what it wrote with check_blackbox() before the ensemble reads it.
"""

import csv
import json
import os

SCORES_ROOT = "scores"  # default parent folder of every black box's output


def predictions_tag(predictions_path):
    """"predictions_rcnn.json" -> "rcnn" (used for names, groups and folders)."""
    stem = os.path.splitext(os.path.basename(predictions_path))[0]
    tag = stem[len("predictions_"):] if stem.startswith("predictions_") else stem
    return "".join(c if c.isalnum() else "_" for c in tag)


def image_paths(labels_path, images_dir):
    """{file_name: absolute path} for every image in labels.json, in file-name order."""
    with open(labels_path) as f:
        images = json.load(f)["images"]
    return {im["file_name"]: os.path.abspath(os.path.join(images_dir, im["file_name"]))
            for im in sorted(images, key=lambda im: im["file_name"])}


def write_blackbox(out_dir, rows, name, method, group, signals, params, weights=None):
    """rows: one dict per image with file_name, file_path, the s_ columns, flagged, reason."""
    os.makedirs(out_dir, exist_ok=True)
    columns = ["file_name", "file_path", *signals, "flagged", "reason"]
    with open(os.path.join(out_dir, "scores.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"name": name, "method": method, "group": group, "signals": signals,
                   "weights": weights or {}, "params": params}, f, indent=2)
    n_flagged = sum(bool(r.get("flagged")) for r in rows)
    print(f"Wrote {out_dir}/scores.csv + meta.json  [{name}: {len(rows)} images, {n_flagged} flagged, "
          f"group '{group}', signals {', '.join(signals)}]")


def check_blackbox(out_dir, file_names):
    """(meta, errors, warnings) for the contract files in out_dir; file_names: the audit's images.
    Errors would make the ensemble fail or silently misread the output; warnings are worth a look."""
    errors, warnings = [], []
    meta_path, scores_path = os.path.join(out_dir, "meta.json"), os.path.join(out_dir, "scores.csv")
    for path in (meta_path, scores_path):
        if not os.path.isfile(path):
            errors.append(f"{os.path.basename(path)} is missing")
    if errors:
        return {}, errors, warnings

    with open(meta_path) as f:
        meta = json.load(f)
    for key in ("name", "method", "group", "signals"):
        if not meta.get(key):
            errors.append(f'meta.json has no "{key}"')
    signals = set(meta.get("signals") or {})
    stray = set(meta.get("weights") or {}) - signals
    if stray:
        warnings.append(f"meta.json weights for unknown signals: {', '.join(sorted(stray))}")

    with open(scores_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        rows = list(reader)
    missing_cols = [c for c in ("file_name", "file_path", "flagged", "reason") if c not in columns]
    if missing_cols:
        errors.append(f"scores.csv has no {', '.join(missing_cols)} column")
    s_cols = [c for c in columns if c.startswith("s_")]
    if not s_cols:
        errors.append("scores.csv has no s_<signal> column")
    if signals - set(s_cols):
        errors.append(f"meta.json signals not in scores.csv: {', '.join(sorted(signals - set(s_cols)))}")
    if set(s_cols) - signals:
        warnings.append(f"scores.csv signals not described in meta.json: {', '.join(sorted(set(s_cols) - signals))}")
    if missing_cols or not s_cols:
        return meta, errors, warnings

    names = [r["file_name"] for r in rows]
    if len(set(names)) < len(names):
        errors.append(f"{len(names) - len(set(names))} duplicate file_name rows")
    expected, have = set(file_names), set(names)
    if not expected & have:
        errors.append("none of its file_name values are images in --labels")
    elif expected - have:
        warnings.append(f"{len(expected - have)} of {len(expected)} images have no row (no opinion)")
    if have - expected:
        warnings.append(f"{len(have - expected)} rows for images not in --labels (ignored)")
    for c in s_cols:
        values = [r[c] for r in rows if r[c] not in ("", None)]
        bad = []
        for v in values:
            try:
                float(v)
            except ValueError:
                bad.append(v)
        if bad:
            errors.append(f"{c} has {len(bad)} non-numeric values, e.g. '{bad[0]}' (blank = no opinion)")
        elif not values:
            warnings.append(f"{c} is blank for every image")
    odd = {r["flagged"] for r in rows} - {"True", "False", "true", "false", ""}
    if odd:
        warnings.append(f"flagged has values other than True / False ({', '.join(sorted(odd))}): read as False")
    return meta, errors, warnings


def read_blackbox(out_dir):
    """(pandas DataFrame indexed by file_name, meta dict)."""
    import pandas as pd

    with open(os.path.join(out_dir, "meta.json")) as f:
        meta = json.load(f)
    df = pd.read_csv(os.path.join(out_dir, "scores.csv"), keep_default_na=True)
    df["flagged"] = df["flagged"].astype(str).str.lower() == "true"
    df["reason"] = df["reason"].fillna("")
    return df.set_index("file_name"), meta
