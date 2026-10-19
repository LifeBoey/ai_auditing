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


def read_blackbox(out_dir):
    """(pandas DataFrame indexed by file_name, meta dict)."""
    import pandas as pd

    with open(os.path.join(out_dir, "meta.json")) as f:
        meta = json.load(f)
    df = pd.read_csv(os.path.join(out_dir, "scores.csv"), keep_default_na=True)
    df["flagged"] = df["flagged"].astype(str).str.lower() == "true"
    df["reason"] = df["reason"].fillna("")
    return df.set_index("file_name"), meta
