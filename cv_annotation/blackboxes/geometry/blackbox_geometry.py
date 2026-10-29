"""
Black box: annotation geometry, judged against the labels themselves (no model).

Every non-crowd box is compared with the other boxes of its class and nothing else, so this
is evidence independent of every model (its own group, "geometry"), and it runs once per audit.
Built from blackbox_template/: prepare() learns each class's typical box size and shape,
score_image() judges one image against that:

    s_size_outlier  the image's most unusual box for its class: robust z-score (median / MAD
                    over the class) of log(box area / image area) or of log(aspect ratio),
                    whichever is larger; blank if none of its classes has --min-class-boxes boxes
    s_degenerate    boxes under --min-side px on their short side, or reaching outside the image
    s_duplicate     pairs of same-class boxes with IoU >= --dup-iou (one object annotated twice)
    flagged         a degenerate or duplicate box, or a size outlier >= --flag-z
    reason          its worst issue, e.g. "size: annotated pizza is 0.05% of the image, aspect 3.10
                    (typical pizza: 9.8%, aspect 1.25; z 6.3)"
Images with no (non-crowd) boxes get no opinion: blank everywhere.

Catches sloppy and broken boxes, double annotations, and wrong classes whose size or shape
doesn't fit the class; it can't see missing annotations (there's no box to measure).

Usage:
    python blackboxes/geometry/blackbox_geometry.py --labels labels.json --images-dir data
    python blackboxes/geometry/blackbox_geometry.py --flag-z 5 --min-side 3
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]  # blackbox_common, run directly from here or blackboxes/<method>/
from blackbox_common import SCORES_ROOT, image_paths, predictions_tag, write_blackbox


# ---------------------------------------------------------------- YOUR CODE ----

METHOD = "geometry"

SIGNALS = {
    "s_size_outlier": "robust z-score of the most unusual box size / aspect for its class",
    "s_degenerate": "count of boxes too thin or reaching outside the image",
    "s_duplicate": "count of same-class box pairs that overlap almost completely",
}

EDGE_TOLERANCE = 1.0  # px a box may stick out of the image before it counts as degenerate
MIN_SCALE = 0.05      # floor on the MAD scale (log units): classes of near-identical boxes don't explode z


def add_options(parser):
    parser.add_argument("--min-side", type=float, default=2.0, help="boxes thinner than this (px) are degenerate")
    parser.add_argument("--dup-iou", type=float, default=0.9, help="same-class boxes overlapping this much are duplicates")
    parser.add_argument("--flag-z", type=float, default=6.0, help="flag a size outlier at this robust z-score")
    parser.add_argument("--min-class-boxes", type=int, default=10,
                        help="classes with fewer boxes get no size opinion (too few to know what's typical)")


def measure(box, image, args):
    """(outside, degenerate, features) of one annotation; features = (log relative area, log aspect),
    None for a degenerate box (its size means nothing)."""
    x, y, w, h = box["bbox"]
    outside = (x < -EDGE_TOLERANCE or y < -EDGE_TOLERANCE
               or x + w > image["width"] + EDGE_TOLERANCE or y + h > image["height"] + EDGE_TOLERANCE)
    degenerate = min(w, h) < args.min_side or outside
    features = None if degenerate else (math.log(w * h / (image["width"] * image["height"])), math.log(w / h))
    return outside, degenerate, features


def iou(a, b):
    """IoU of two COCO [x, y, w, h] boxes."""
    iw = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    ih = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    inter = max(iw, 0) * max(ih, 0)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def prepare(images, annotations, predictions, args):
    """{class: (median, scale) per feature} for classes with at least --min-class-boxes measurable
    boxes; scale = 1.4826 * MAD, which matches the standard deviation for normal data."""
    by_class = defaultdict(list)
    for name, boxes in annotations.items():
        for b in boxes:
            if not b.get("iscrowd"):
                features = measure(b, images[name], args)[2]
                if features is not None:
                    by_class[b["class_name"]].append(features)
    stats = {}
    for cls, feats in by_class.items():
        if len(feats) >= args.min_class_boxes:
            f = np.array(feats)
            med = np.median(f, axis=0)
            stats[cls] = (med, np.maximum(1.4826 * np.median(np.abs(f - med), axis=0), MIN_SCALE))
    return stats


def score_image(image, annotations, predictions, context, args):
    stats = context
    boxes = [b for b in annotations if not b.get("iscrowd")]  # crowd regions are big by design
    if not boxes:
        return {s: None for s in SIGNALS}, False, ""
    issues = []  # (priority, text): degenerate, then duplicate, then size

    degenerate, worst = 0, None  # worst: (z, box, features)
    for b in boxes:
        outside, is_degenerate, features = measure(b, image, args)
        if is_degenerate:
            degenerate += 1
            w, h = b["bbox"][2:]
            issues.append((0, f"degenerate: annotated {b['class_name']} box {w:.1f}x{h:.1f}px"
                              + (" reaching outside the image" if outside else "")))
        elif b["class_name"] in stats:
            med, scale = stats[b["class_name"]]
            z = float(np.max(np.abs(np.array(features) - med) / scale))
            if worst is None or z > worst[0]:
                worst = (z, b, features)

    duplicates = 0
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            if a["class_name"] == b["class_name"] and (overlap := iou(a["bbox"], b["bbox"])) >= args.dup_iou:
                duplicates += 1
                issues.append((1, f"duplicate: two {a['class_name']} boxes overlap (IoU {overlap:.2f})"))

    if worst and worst[0] >= args.flag_z:
        z, b, features = worst
        med, _ = stats[b["class_name"]]
        issues.append((2, f"size: annotated {b['class_name']} is {100 * math.exp(features[0]):.2g}% of the image, "
                          f"aspect {math.exp(features[1]):.2f} (typical {b['class_name']}: "
                          f"{100 * math.exp(med[0]):.2g}%, aspect {math.exp(med[1]):.2f}; z {z:.1f})"))

    issues.sort(key=lambda i: i[0])
    scores = {"s_size_outlier": round(worst[0], 4) if worst else None,
              "s_degenerate": degenerate, "s_duplicate": duplicates}
    reason = issues[0][1] + (f" (+{len(issues) - 1} more)" if len(issues) > 1 else "") if issues else ""
    return scores, bool(issues), reason


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
