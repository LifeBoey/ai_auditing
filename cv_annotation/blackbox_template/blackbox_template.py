"""
Template black box. You only write score_image(): it gets one image's annotations (and, if you
want, one model's predictions for it) and says how suspicious they look. Everything else (reading
the files, the per-image loop, writing scores.csv + meta.json) is done below and can stay as it is.

To make your own:
  1. cp -r blackbox_template blackboxes/<your_method>
  2. in blackboxes/<your_method>/: rename this file to blackbox_<your_method>.py, and in
     blackbox_config.json set "script" to that name and fill in "description"
  3. fill in the TODOs below (the example code shows what each one looks like)
  4. python run_audit.py --list-methods                       # is it found?
     python run_audit.py --methods <your_method> --labels ... --images-dir ... --evaluate-planted
                                                             # does it catch the planted errors?
  5. delete "to_run": false from blackbox_config.json once it's good enough to run by default
More in blackboxes/README.md.

The example below scores two things:
    s_tiny_boxes            annotated boxes under --min-side px on their short side
    s_class_not_predicted   annotated classes the model found nowhere in the image
                            (only with "predictions": "per_model" in blackbox_config.json)
"""

import argparse
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]  # blackbox_common, run directly from here or blackboxes/<method>/
from blackbox_common import SCORES_ROOT, image_paths, predictions_tag, write_blackbox


# ---------------------------------------------------------------- YOUR CODE ----

# TODO 1: your method's name: used in meta.json, the ensemble's columns and --weights
METHOD = "template"

# TODO 2: every score you produce, with one line saying what it measures.
# Names start with s_. HIGHER = MORE SUSPICIOUS; any scale (counts, probabilities, ...).
SIGNALS = {
    "s_tiny_boxes": "annotated boxes thinner than --min-side px",
    "s_class_not_predicted": "annotated classes the model predicted nowhere in the image",
}


# TODO 3: your own options (thresholds etc.), with defaults that work.
# Change them per run with: run_audit.py --box-args <your_method>="--min-side 4"
def add_options(parser):
    parser.add_argument("--min-side", type=float, default=2.0, help="boxes thinner than this (px) are tiny")


# TODO 4 (optional): work out anything that needs the whole dataset at once (class frequencies,
# typical box sizes, ...). Runs once, before score_image(); what it returns is score_image's `context`.
def prepare(images, annotations, predictions, args):
    """images: {file_name: labels.json entry}; annotations / predictions: {file_name: [boxes]} as
    score_image() gets them (predictions: None without a model)."""
    return None


# TODO 5: judge one image.
def score_image(image, annotations, predictions, context, args):
    """
    image        its entry in labels.json: {"file_name", "width", "height", ...}
    annotations  its boxes: [{"class_name", "bbox": [x, y, w, h], ...}]  (bbox: top-left corner, width, height)
    predictions  the model's boxes: [{"class_name", "score", "bbox": [x1, y1, x2, y2]}]  (bbox: two corners)
                 None if blackbox_config.json says "predictions": "none"
    context      whatever prepare() returned
    args         the parsed command line, including your options from add_options()

    Returns (scores, flagged, reason):
      scores   {signal: number}, one entry per SIGNALS key. None = "no opinion" on this image
               (it then counts neither for nor against it); 0 = "looked, found nothing wrong".
      flagged  True if your method calls this image a problem on its own
      reason   short text on its worst problem, shown on the review list ("" if none)
    """
    tiny = [a for a in annotations if min(a["bbox"][2], a["bbox"][3]) < args.min_side]

    if predictions is None:
        not_predicted = None  # no model to compare with: no opinion
    else:
        predicted = {p["class_name"] for p in predictions}
        not_predicted = sorted({a["class_name"] for a in annotations} - predicted)

    scores = {
        "s_tiny_boxes": len(tiny),
        "s_class_not_predicted": None if not_predicted is None else len(not_predicted),
    }
    if tiny:
        w, h = tiny[0]["bbox"][2:]
        return scores, True, f"tiny box: annotated {tiny[0]['class_name']} is {w:.1f}x{h:.1f}px"
    if not_predicted:
        return scores, True, f"model sees no {', '.join(not_predicted)} here"
    return scores, False, ""


# ------------------------------------------------- PLUMBING: no need to edit ----

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", default="labels.json")
    parser.add_argument("--images-dir", default="data")
    parser.add_argument("--predictions", default=None, help="one predictions file (per_model black boxes)")
    parser.add_argument("--out-dir", default=None, help=f"black-box output (default: {SCORES_ROOT}/<name>)")
    add_options(parser)
    args = parser.parse_args()

    with open(args.labels) as f:
        coco = json.load(f)
    class_names = {c["id"]: c["name"] for c in coco["categories"]}
    images = {im["file_name"]: im for im in coco["images"]}
    file_names = {im["id"]: im["file_name"] for im in coco["images"]}
    annotations = defaultdict(list)
    for a in coco["annotations"]:
        annotations[file_names[a["image_id"]]].append({**a, "class_name": class_names[a["category_id"]]})

    predictions = None
    if args.predictions:
        with open(args.predictions) as f:
            predictions = {item["file_name"]: item["prediction"] for item in json.load(f)}

    paths = image_paths(args.labels, args.images_dir)
    if predictions is not None:
        predictions = {name: predictions.get(name, []) for name in paths}
    context = prepare(images, {name: annotations[name] for name in paths}, predictions, args)

    rows = []
    for name, path in paths.items():
        preds = None if predictions is None else predictions[name]
        scores, flagged, reason = score_image(images[name], annotations[name], preds, context, args)
        missing = set(SIGNALS) - set(scores)
        if missing:
            raise SystemExit(f"score_image() returned no {', '.join(sorted(missing))} for {name} (use None for no opinion)")
        rows.append({"file_name": name, "file_path": path, "flagged": bool(flagged), "reason": reason,
                     **{s: "" if v is None else v for s, v in scores.items()}})

    # one model's predictions: named and grouped after that model (correlated with the other
    # methods using it); no model: its own name and its own group (independent evidence)
    tag = predictions_tag(args.predictions) if args.predictions else None
    name = f"{METHOD}_{tag}" if tag else METHOD
    write_blackbox(args.out_dir or os.path.join(SCORES_ROOT, name), rows, name=name, method=METHOD,
                   group=tag or METHOD, signals=SIGNALS,
                   params={k: v for k, v in vars(args).items() if k != "out_dir"})


if __name__ == "__main__":
    main()
