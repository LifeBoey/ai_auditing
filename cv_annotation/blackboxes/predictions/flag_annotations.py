"""
Flag images whose ANNOTATIONS look wrong or incomplete, using one model's predictions.
(Annotation quality only: image quality and duplicates are the audit scripts' job.)

Every annotated box is compared with what the model predicted around it, and every rule
says in plain words why it fired, e.g. "annotated dog; model says cat 0.91 at IoU 0.83".

  Per annotated box (at most one of these, checked in this order):
    wrong_class     a confident prediction of ANOTHER class sits on the box (IoU >= --match-iou)
                    and beats the annotated class by --swap-margin
    bad_location    the model is confident about this class nearby (--loc-iou <= IoU < --match-iou)
                    but not on the box itself: loose, shifted or cut-off box
    unsupported     no prediction of any class above --support-score overlaps it (IoU >= --support-iou):
                    the model sees nothing there (extra / hallucinated box)
    low_confidence  right class, but its best score is below --low-conf. WEAK: adds to the
                    image score but never flags an image on its own (often just a hard object)
  Per annotated box, independent of the above:
    duplicate_box   two annotations of the same class with IoU >= --dup-iou
    invalid_box     smaller than --min-box-side px, or outside the image
  Per prediction:
    missing         a prediction above --missing-conf that overlaps no annotation of any class
                    (IoU < --match-iou): an object nobody annotated

Each flag has a severity in [0, 1] (unsupported and missing are scaled down for objects
smaller than --full-size, which detectors miss all the time; unsupported is also capped at
--unsupported-cap and divided by sqrt(annotations of that class in the image)). An image's score is 1 - prod(1 - severity) over its flags,
and an image is flagged if it has a flag other than low_confidence with severity >= --flag-severity.

--calibrate replaces the fixed "confident" thresholds with per-class ones (the median score of
that class's correct matches), since the model is far more confident on "person" than "spoon".

--evaluate-planted (test sets only, off by default): planted test errors ("<name>_wrong.jpg" /
"<name>_missing.jpg", annotations changed from "<name>.jpg") are read from labels.json, marked in
the CSVs and the summary reports how many the rules catch. --sweep grid-searches the wrong_class
thresholds against them. Without it, file names mean nothing.

Inputs:
    labels.json             COCO ground truth (bbox = [x, y, w, h])
    predictions_*.json      [{"file_name", "prediction": [{"class_id", "class_name", "score",
                              "bbox": [x1, y1, x2, y2]}]}]  (use a low-threshold file: the
                              unsupported / low_confidence rules need the low scores)
Outputs (in --out-dir):
    flagged_images.csv      flagged images, worst first: path, score, main issue, per-flag counts
    flagged_boxes.csv       one row per flag with the evidence; fix boxes from this list
    all_images.csv          every image with its score and max_severity (worst single flag)
    viz/                    annotations (green, flagged ones orange) vs the predictions that
                              triggered a flag (red) for the --num-viz worst images

Usage:
    python flag_annotations.py
    python flag_annotations.py --predictions predictions_yolo.json --out-dir flags_yolo
    python flag_annotations.py --calibrate --evaluate-planted --sweep
"""

import argparse
import csv
import itertools
import json
import os
from collections import Counter, defaultdict

import numpy as np
from PIL import Image, ImageDraw

BOX_FLAGS = ["wrong_class", "bad_location", "unsupported", "low_confidence"]  # one per box, in priority order
FLAGS = BOX_FLAGS + ["duplicate_box", "invalid_box", "missing"]
WEAK_FLAGS = {"low_confidence"}  # count toward the image score, never flag an image alone


# ============================================================
# Loading
# ============================================================

def xywh_to_xyxy(b):
    x, y, w, h = b
    return [x, y, x + w, y + h]


def iou_matrix(a, b):
    """IoU between every box in a (N,4) and b (M,4), xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def inside_fraction(boxes, regions):
    """For each box, the max fraction of its area covered by any region."""
    if len(boxes) == 0 or len(regions) == 0:
        return np.zeros(len(boxes))
    x1 = np.maximum(boxes[:, None, 0], regions[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], regions[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], regions[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], regions[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return (inter / (area[:, None] + 1e-9)).max(axis=1)


def load_data(coco, predictions_path, images_dir, crowd_ignore, min_score):
    """One dict per image with its non-crowd annotations and predictions as arrays."""
    with open(predictions_path) as f:
        preds_by_file = {item["file_name"]: item["prediction"] for item in json.load(f)}

    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    data, dropped_crowd, no_preds = [], 0, 0
    for image in sorted(coco["images"], key=lambda im: im["file_name"]):
        anns = [a for a in anns_by_image[image["id"]] if not a.get("iscrowd", 0)]
        crowd = [a for a in anns_by_image[image["id"]] if a.get("iscrowd", 0)]
        if image["file_name"] not in preds_by_file:
            no_preds += 1
        raw = [p for p in preds_by_file.get(image["file_name"], []) if p["score"] >= min_score]

        # Crowd regions aren't single-object labels: drop same-class predictions inside them
        # so they aren't reported as missing annotations.
        if raw and crowd and crowd_ignore < 1.0:
            keep = np.ones(len(raw), dtype=bool)
            for cat_id in {a["category_id"] for a in crowd}:
                regions = np.array([xywh_to_xyxy(a["bbox"]) for a in crowd if a["category_id"] == cat_id])
                idx = [n for n, p in enumerate(raw) if int(p["class_id"]) == cat_id]
                if idx:
                    boxes = np.array([raw[n]["bbox"] for n in idx], dtype=float)
                    keep[np.array(idx)[inside_fraction(boxes, regions) >= crowd_ignore]] = False
            dropped_crowd += int((~keep).sum())
            raw = [p for p, k in zip(raw, keep) if k]

        data.append({
            "image": image,
            "path": os.path.abspath(os.path.join(images_dir, image["file_name"])),
            "gt_ids": [a["id"] for a in anns],
            "gt_boxes": np.array([xywh_to_xyxy(a["bbox"]) for a in anns], dtype=float).reshape(-1, 4),
            "gt_cls": np.array([a["category_id"] for a in anns], dtype=int),
            "pred_boxes": np.array([p["bbox"] for p in raw], dtype=float).reshape(-1, 4),
            "pred_cls": np.array([int(p["class_id"]) for p in raw], dtype=int),
            "pred_scores": np.array([p["score"] for p in raw], dtype=float),
        })

    print(f"Images: {len(data)} ({no_preds} missing from the predictions file) | "
          f"annotations (non-crowd): {sum(len(d['gt_cls']) for d in data)} | "
          f"predictions >= {min_score}: {sum(len(d['pred_cls']) for d in data)} "
          f"(dropped {dropped_crowd} inside crowd regions)")
    return data


def find_planted(coco):
    """{file_name: "wrong" | "duplicate"} for the planted images, and the ids of the planted
    wrong-class annotations (boxes in a *_wrong image whose class differs from the same box in
    the original). None if nothing was planted."""
    by_name = {im["file_name"]: im for im in coco["images"]}
    anns_by_image = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_image[a["image_id"]].append(a)
    planted_images, planted_anns = {}, set()
    for name, image in by_name.items():
        stem, ext = os.path.splitext(name)
        for suffix, kind in (("_wrong", "wrong"), ("_duplicate", "duplicate"), ("_missing", "missing")):
            original = by_name.get(stem[:-len(suffix)] + ext) if stem.endswith(suffix) else None
            if original is None:
                continue
            planted_images[name] = kind
            if kind == "wrong":
                orig = {tuple(a["bbox"]): a["category_id"] for a in anns_by_image[original["id"]]}
                planted_anns |= {a["id"] for a in anns_by_image[image["id"]]
                                 if orig.get(tuple(a["bbox"])) != a["category_id"]}
    return (planted_images, planted_anns) if planted_images else None


# ============================================================
# Rules
# ============================================================

def class_thresholds(data, args):
    """Score at which a prediction counts as confident (and as a missing object), per class.
    Fixed unless --calibrate: then the median (75th percentile for missing) score of that
    class's predictions that correctly match an annotation, clipped to [--calib-floor, fixed]."""
    conf = defaultdict(lambda: args.conf)
    missing_conf = defaultdict(lambda: args.missing_conf)
    if not args.calibrate:
        return conf, missing_conf
    matched = defaultdict(list)
    for d in data:
        iou = iou_matrix(d["gt_boxes"], d["pred_boxes"])
        hit = (iou >= args.match_iou) & (d["gt_cls"][:, None] == d["pred_cls"][None, :])
        for g in np.flatnonzero(hit.any(axis=1)):
            matched[d["gt_cls"][g]].append(d["pred_scores"][hit[g]].max())
    for c, s in matched.items():
        conf[c] = float(np.clip(np.median(s), args.calib_floor, args.conf))
        missing_conf[c] = float(np.clip(np.percentile(s, 75), args.calib_floor, args.missing_conf))
    return conf, missing_conf


def flag_image(d, args, conf, missing_conf):
    """All flags for one image, as dicts: flag, severity, g (annotation index), p (prediction
    index), iou, plus extra evidence."""
    G, gc = d["gt_boxes"], d["gt_cls"]
    P, pc, ps = d["pred_boxes"], d["pred_cls"], d["pred_scores"]
    n, m = len(gc), len(pc)
    W, H = d["image"]["width"], d["image"]["height"]
    iou = iou_matrix(G, P) if n and m else np.zeros((n, m))
    same = gc[:, None] == pc[None, :]
    pconf = np.array([conf[c] for c in pc])  # confident threshold of each prediction's class
    flags, badloc_preds = [], set()

    def add(flag, severity, g=None, p=None, iou_value=None, **extra):
        flags.append({"flag": flag, "severity": float(np.clip(severity, 0, 1)), "g": g, "p": p,
                      "iou": None if iou_value is None else float(iou_value), **extra})

    def best(mask):
        """Index of the highest-scoring prediction in mask, or None."""
        return int(np.flatnonzero(mask)[np.argmax(ps[mask])]) if mask.any() else None

    def size_weight(box):
        """Detectors miss small objects all the time, so "the model sees nothing / something
        unannotated" means less for them: sqrt(area) / --full-size, capped at 1."""
        return min(1.0, float(np.sqrt(max((box[2] - box[0]) * (box[3] - box[1]), 0))) / args.full_size)

    side = np.minimum(G[:, 2] - G[:, 0], G[:, 3] - G[:, 1])
    invalid = ((side < args.min_box_side) | (G[:, 0] < -2) | (G[:, 1] < -2)
               | (G[:, 2] > W + 2) | (G[:, 3] > H + 2))

    for g in range(n):
        if invalid[g]:
            add("invalid_box", 1.0, g, side=float(side[g]))
            continue
        row = iou[g]
        at_match = row >= args.match_iou
        p_same = best(at_match & same[g])
        s_match = ps[p_same] if p_same is not None else 0.0

        # wrong class: a confident other-class prediction on the same object beats the annotated class
        p = best(at_match & ~same[g] & (ps >= pconf))
        if p is not None and ps[p] - s_match >= args.swap_margin:
            add("wrong_class", ps[p] - s_match, g, p, row[p], same_score=s_match)
            continue

        # bad location: confident about this class nearby, but not on this box
        if s_match < conf[gc[g]]:
            p = best(same[g] & (row >= args.loc_iou) & ~at_match & (ps >= pconf))
            if p is not None:
                badloc_preds.add(p)
                add("bad_location", ps[p] * (1 - row[p]), g, p, row[p], same_score=s_match)
                continue

        # unsupported: nothing of any class is predicted here
        p = best(row >= args.support_iou)
        support = ps[p] if p is not None else 0.0
        # Absence of a prediction is weaker evidence than a contradicting one (capped), and
        # weaker still in busy scenes, where the detector drops one of many similar objects.
        if support < args.support_score:
            crowd = 1 / np.sqrt((gc == gc[g]).sum())
            add("unsupported", args.unsupported_cap * (1 - support / args.support_score) * size_weight(G[g]) * crowd,
                g, p, row[p] if p is not None else None, same_score=s_match)
            continue

        # low confidence: right class, weakly
        if s_match < args.low_conf:
            add("low_confidence", args.weak_cap * (1 - s_match / args.low_conf), g, p_same,
                row[p_same] if p_same is not None else None, same_score=s_match)

    # duplicate boxes: flag the later annotation of each same-class pair
    if n > 1:
        gg = iou_matrix(G, G)
        for b in range(1, n):
            a = [a for a in range(b) if gc[a] == gc[b] and gg[a, b] >= args.dup_iou]
            if a:
                add("duplicate_box", gg[a[0], b], b, None, gg[a[0], b], duplicate_of=d["gt_ids"][a[0]])

    # missing: confident predictions overlapping no annotation; one flag per object
    if m:
        max_iou = iou.max(axis=0) if n else np.zeros(m)
        kept = []
        for p in np.argsort(-ps):
            if ps[p] < missing_conf[pc[p]] or max_iou[p] >= args.match_iou or p in badloc_preds:
                continue
            if kept and iou_matrix(P[[p]], P[kept]).max() >= args.match_iou:
                continue  # the same object, predicted twice
            kept.append(p)
            add("missing", ps[p] * size_weight(P[p]), None, int(p), max_iou[p])
    return flags


def match_counts(d, args):
    """(tp, fp, fn): greedy same-class matching at --match-iou of predictions >= --eval-conf."""
    keep = np.flatnonzero(d["pred_scores"] >= args.eval_conf)
    keep = keep[np.argsort(-d["pred_scores"][keep])]
    iou = iou_matrix(d["gt_boxes"], d["pred_boxes"][keep])
    used, tp = set(), 0
    for j, p in enumerate(keep):
        cand = [(iou[g, j], g) for g in range(len(d["gt_cls"]))
                if g not in used and d["gt_cls"][g] == d["pred_cls"][p] and iou[g, j] >= args.match_iou]
        if cand:
            used.add(max(cand)[1])
            tp += 1
    return tp, len(keep) - tp, len(d["gt_cls"]) - tp


def run_rules(data, args):
    conf, missing_conf = class_thresholds(data, args)
    return [flag_image(d, args, conf, missing_conf) for d in data]


# ============================================================
# Outputs
# ============================================================

def fmt_box(b):
    """xyxy -> "x,y,w,h" (COCO layout, like labels.json)."""
    x1, y1, x2, y2 = b
    return f"{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f}"


def reason(f, d, names):
    g, p = f["g"], f["p"]
    gt = names.get(d["gt_cls"][g], "?") if g is not None else None
    pred = f"{names.get(d['pred_cls'][p], '?')} {d['pred_scores'][p]:.2f}" if p is not None else None
    at = f" at IoU {f['iou']:.2f}" if f["iou"] is not None else ""
    if f["flag"] == "wrong_class":
        return f"annotated {gt}; model says {pred}{at} ({gt} {f['same_score']:.2f})"
    if f["flag"] == "bad_location":
        return f"annotated {gt}; best confident {gt} box only overlaps{at} (score {pred.split()[-1]})"
    if f["flag"] == "unsupported":
        return f"annotated {gt}; nothing predicted there" + (f" (best: {pred}{at})" if pred else "")
    if f["flag"] == "low_confidence":
        return f"annotated {gt}; model only gives {gt} {f['same_score']:.2f}"
    if f["flag"] == "duplicate_box":
        return f"annotated {gt} twice{at} (same as annotation {f['duplicate_of']})"
    if f["flag"] == "invalid_box":
        return f"annotated {gt}; box {f['side']:.1f}px on its short side or outside the image"
    return f"model sees {pred} where nothing is annotated (best overlap with an annotation: IoU {f['iou']:.2f})"


def write_csv(path, rows):
    if not rows:
        print(f"(nothing to write to {path})")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def draw(d, flags, names, path):
    im = Image.open(d["path"]).convert("RGB")
    dr = ImageDraw.Draw(im)
    flagged = {f["g"]: f["flag"] for f in flags if f["g"] is not None}
    for g, box in enumerate(d["gt_boxes"]):
        color = "orange" if g in flagged else "lime"
        dr.rectangle(box.tolist(), outline=color, width=3 if g in flagged else 1)
        dr.text((box[0] + 2, box[1] + 2), names.get(d["gt_cls"][g], "?") +
                (f" [{flagged[g]}]" if g in flagged else ""), fill=color)
    for p in {f["p"] for f in flags if f["p"] is not None}:
        box = d["pred_boxes"][p]
        dr.rectangle(box.tolist(), outline="red", width=2)
        dr.text((box[0] + 2, box[3] - 12), f"{names.get(d['pred_cls'][p], '?')} {d['pred_scores'][p]:.2f}", fill="red")
    im.save(path)


def image_score(flags):
    return 1 - float(np.prod([1 - f["severity"] for f in flags])) if flags else 0.0


# ============================================================
# Planted-error check
# ============================================================

def planted_report(data, all_flags, planted, image_rows, flag_severity):
    planted_images, planted_anns = planted
    box_flag = {}  # annotation id -> (flag, severity)
    for d, flags in zip(data, all_flags):
        for f in flags:
            if f["g"] is not None and f["flag"] in BOX_FLAGS:
                box_flag[d["gt_ids"][f["g"]]] = (f["flag"], f["severity"])
    scored = planted_anns & {a for d in data for a in d["gt_ids"]}
    print(f"\nPlanted wrong-class boxes: {len(scored)} (+{len(planted_anns) - len(scored)} crowd, not checked)")
    by_flag = Counter(box_flag[a][0] for a in scored if a in box_flag)
    for flag in BOX_FLAGS:
        total = sum(v[0] == flag for v in box_flag.values())
        print(f"  {flag:15s} {by_flag[flag]:4d} planted of {total:5d} flagged")
    strong = [(s, a in planted_anns) for a, (fl, s) in box_flag.items()
              if fl not in WEAK_FLAGS and s >= flag_severity]
    hit = sum(p for _, p in strong)
    print(f"  any flag but low_confidence, severity >= {flag_severity}: {hit} / {len(scored)} ({hit / max(len(scored), 1):.0%} recall)")
    strong.sort(key=lambda x: -x[0])
    for k in (250, 500, 1000, 2000):
        h = sum(p for _, p in strong[:k])
        print(f"  top {k:4d} boxes by severity: {h:4d} planted ({h / max(len(scored), 1):4.0%} recall, {h / k:4.0%} precision)")

    ranked = [r["file_name"] for r in image_rows]  # sorted worst first
    for kind in ("wrong", "missing"):
        rows = [r for r in image_rows if planted_images.get(r["file_name"]) == kind]
        if not rows:
            continue
        print(f"  _{kind} images flagged: {sum(r['flagged'] for r in rows)} / {len(rows)}  "
              f"(main issue: {dict(Counter(r['main_issue'] for r in rows).most_common(3))})")
        for k in (100, 200, 500):
            print(f"  _{kind} images in the worst {k:3d}: {sum(planted_images.get(n) == kind for n in ranked[:k])}")


def sweep(data, args, planted):
    """Grid-search the wrong_class thresholds against the planted wrong-class boxes. Precision
    counts every unplanted flag as a false alarm, so real COCO mistakes lower it."""
    _, planted_anns = planted
    n_planted = len(planted_anns & {a for d in data for a in d["gt_ids"]})
    print("\n=== Sweep: wrong_class vs planted boxes ===")
    print(f"{'match_iou':>9} {'conf':>5} {'margin':>6} | {'flagged':>7} {'planted':>7} {'recall':>6} {'precision':>9} {'F1':>5}")
    results = []
    for match_iou, conf, margin in itertools.product((0.3, 0.5, 0.7), (0.3, 0.5, 0.7), (0.1, 0.3, 0.5)):
        a = argparse.Namespace(**{**vars(args), "match_iou": match_iou, "conf": conf, "swap_margin": margin})
        flagged = planted_hit = 0
        for d, flags in zip(data, run_rules(data, a)):
            for f in flags:
                if f["flag"] == "wrong_class":
                    flagged += 1
                    planted_hit += d["gt_ids"][f["g"]] in planted_anns
        r, p = planted_hit / max(n_planted, 1), planted_hit / max(flagged, 1)
        results.append((2 * p * r / max(p + r, 1e-9), match_iou, conf, margin, flagged, planted_hit, r, p))
    for f1, match_iou, conf, margin, flagged, hit, r, p in sorted(results, reverse=True):
        print(f"{match_iou:9.1f} {conf:5.1f} {margin:6.1f} | {flagged:7d} {hit:7d} {r:6.0%} {p:9.0%} {f1:5.2f}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", default="labels.json")
    parser.add_argument("--predictions", default="predictions_rcnn.json")
    parser.add_argument("--images-dir", default="data")
    parser.add_argument("--out-dir", default=None, help="default: flags_<predictions name>")
    parser.add_argument("--min-score", type=float, default=0.01, help="ignore predictions below this score")
    parser.add_argument("--crowd-ignore", type=float, default=0.5,
                        help="drop predictions >= this fraction inside a same-class crowd region; 1.0 disables")
    parser.add_argument("--match-iou", type=float, default=0.5, help="IoU at which a prediction is on the same object")
    parser.add_argument("--conf", type=float, default=0.5, help="score at which a prediction counts as confident")
    parser.add_argument("--swap-margin", type=float, default=0.5,  # best F1 in --sweep on rcnn
                        help="wrong_class: other-class score minus annotated-class score needed")
    parser.add_argument("--loc-iou", type=float, default=0.1, help="bad_location: lowest IoU still counted as nearby")
    parser.add_argument("--support-iou", type=float, default=0.3, help="unsupported: overlap needed to count as support")
    parser.add_argument("--support-score", type=float, default=0.1, help="unsupported: support below this score")
    parser.add_argument("--unsupported-cap", type=float, default=0.7,
                        help="unsupported: max severity (below a typical wrong_class; 1 = old behaviour)")
    parser.add_argument("--low-conf", type=float, default=0.3, help="low_confidence: annotated-class score below this")
    parser.add_argument("--weak-cap", type=float, default=0.3, help="max severity of a low_confidence flag")
    parser.add_argument("--missing-conf", type=float, default=0.7, help="missing: prediction score needed")
    parser.add_argument("--full-size", type=float, default=64,
                        help="unsupported / missing: objects smaller than this (sqrt of area, px) get a lower severity")
    parser.add_argument("--dup-iou", type=float, default=0.9, help="duplicate_box: IoU between same-class annotations")
    parser.add_argument("--min-box-side", type=float, default=2.0, help="invalid_box: shortest side below this (px)")
    parser.add_argument("--flag-severity", type=float, default=0.9,
                        help="an image is flagged if it has a non-weak flag at least this severe")
    parser.add_argument("--eval-conf", type=float, default=0.5, help="score cutoff for the TP/FP/FN counts")
    parser.add_argument("--calibrate", action="store_true", help="per-class confident thresholds (see top)")
    parser.add_argument("--calib-floor", type=float, default=0.3, help="lowest per-class threshold with --calibrate")
    parser.add_argument("--evaluate-planted", action="store_true",
                        help="test sets only: mark and report planted *_wrong / *_missing errors (default: off)")
    parser.add_argument("--sweep", action="store_true",
                        help="grid-search wrong_class thresholds on the planted errors (needs --evaluate-planted)")
    parser.add_argument("--num-viz", type=int, default=20, help="draw the N worst flagged images")
    args = parser.parse_args()
    if args.sweep and not args.evaluate_planted:
        parser.error("--sweep tunes against planted errors: add --evaluate-planted")

    out_dir = args.out_dir or "flags_" + os.path.splitext(os.path.basename(args.predictions))[0].replace("predictions_", "")
    os.makedirs(out_dir, exist_ok=True)

    with open(args.labels) as f:
        coco = json.load(f)
    names = {c["id"]: c["name"] for c in coco["categories"]}
    planted = find_planted(coco) if args.evaluate_planted else None
    data = load_data(coco, args.predictions, args.images_dir, args.crowd_ignore, args.min_score)
    all_flags = run_rules(data, args)

    # --- per-flag rows, worst first ---
    box_rows = []
    for d, flags in zip(data, all_flags):
        for f in flags:
            g, p = f["g"], f["p"]
            row = {
                "file_path": d["path"], "file_name": d["image"]["file_name"], "image_id": d["image"]["id"],
                "flag": f["flag"], "severity": round(f["severity"], 4), "reason": reason(f, d, names),
                "annotation_id": d["gt_ids"][g] if g is not None else "",
                "gt_class": names.get(d["gt_cls"][g], "") if g is not None else "",
                "gt_bbox_xywh": fmt_box(d["gt_boxes"][g]) if g is not None else "",
                "pred_class": names.get(d["pred_cls"][p], "") if p is not None else "",
                "pred_score": round(float(d["pred_scores"][p]), 4) if p is not None else "",
                "pred_bbox_xywh": fmt_box(d["pred_boxes"][p]) if p is not None else "",
                "iou": round(f["iou"], 4) if f["iou"] is not None else "",
            }
            if planted:
                row["is_planted_error"] = row["annotation_id"] in planted[1] if g is not None else ""
            box_rows.append(row)
    box_rows.sort(key=lambda r: -r["severity"])

    # --- per-image rows, worst first ---
    image_rows = []
    for d, flags in zip(data, all_flags):
        counts = Counter(f["flag"] for f in flags)
        main_flag = max(flags, key=lambda f: f["severity"]) if flags else None
        tp, fp, fn = match_counts(d, args)
        row = {
            "file_path": d["path"], "file_name": d["image"]["file_name"], "image_id": d["image"]["id"],
            "score": round(image_score(flags), 4),
            # worst single flag: unlike score, doesn't grow with the number of objects
            "max_severity": round(max((f["severity"] for f in flags if f["flag"] not in WEAK_FLAGS), default=0.0), 4),
            "flagged": any(f["flag"] not in WEAK_FLAGS and f["severity"] >= args.flag_severity for f in flags),
            "main_issue": main_flag["flag"] if main_flag else "",
            "main_reason": reason(main_flag, d, names) if main_flag else "",
            **{f"n_{flag}": counts[flag] for flag in FLAGS},
            "n_annotations": len(d["gt_cls"]), "n_predictions": len(d["pred_cls"]),
            "eval_tp": tp, "eval_fp": fp, "eval_fn": fn,
        }
        if planted:
            row["planted"] = planted[0].get(d["image"]["file_name"], "")
        image_rows.append(row)
    image_rows.sort(key=lambda r: (-r["flagged"], -r["score"]))

    write_csv(os.path.join(out_dir, "flagged_images.csv"), [r for r in image_rows if r["flagged"]])
    write_csv(os.path.join(out_dir, "flagged_boxes.csv"), box_rows)
    write_csv(os.path.join(out_dir, "all_images.csv"), image_rows)

    if args.num_viz:
        viz_dir = os.path.join(out_dir, "viz")
        os.makedirs(viz_dir, exist_ok=True)
        by_name = {d["image"]["file_name"]: (d, fl) for d, fl in zip(data, all_flags)}
        for rank, r in enumerate([r for r in image_rows if r["flagged"]][:args.num_viz], 1):
            d, fl = by_name[r["file_name"]]
            draw(d, fl, names, os.path.join(viz_dir, f"{rank:03d}_{r['file_name']}"))

    # --- summary ---
    print("=" * 60)
    print(f"Predictions: {args.predictions}" + ("  (per-class calibrated thresholds)" if args.calibrate else ""))
    flagged = [r for r in image_rows if r["flagged"]]
    print(f"Images flagged: {len(flagged)} / {len(image_rows)}")
    for flag in FLAGS:
        n_boxes = sum(r[f"n_{flag}"] for r in image_rows)
        n_images = sum(r[f"n_{flag}"] > 0 for r in image_rows)
        print(f"  {flag:15s} {n_boxes:6d} {'predictions' if flag == 'missing' else 'boxes':11s} in {n_images:5d} images"
              + ("  (weak, doesn't flag an image alone)" if flag in WEAK_FLAGS else ""))
    print(f"Match counts (IoU>={args.match_iou}, conf>={args.eval_conf}): TP={sum(r['eval_tp'] for r in image_rows)} "
          f"FP={sum(r['eval_fp'] for r in image_rows)} FN={sum(r['eval_fn'] for r in image_rows)}")
    if planted:
        planted_report(data, all_flags, planted, image_rows, args.flag_severity)
        if args.sweep:
            sweep(data, args, planted)
    print("=" * 60)
    print("Worst 10 images:")
    for r in flagged[:10]:
        print(f"  {r['score']:.3f}  {r['file_path']}\n         {r['main_issue']}: {r['main_reason']}")
    print(f"\nWrote {out_dir}/flagged_images.csv, flagged_boxes.csv, all_images.csv"
          + (f", viz/ ({min(args.num_viz, len(flagged))} images)" if args.num_viz else ""))


if __name__ == "__main__":
    main()
