"""
Audit a COCO dataset with the cleanlab ecosystem.

  A. QUALITY     -> which images are good enough to bother annotating
                    (CleanVision: dark, light, blurry, low_information, odd_aspect_ratio,
                    odd_size, grayscale; plus unreadable files and images that are too small)
  B. DUPLICATES  -> near-duplicate images you shouldn't pay to annotate twice
                    (CLIP embeddings + cleanlab Datalab near_duplicate / outlier)
  C. ANNOTATIONS -> existing labels that are likely missing / wrong / badly placed
                    (cleanlab ObjectLab, using an independent detector's predictions)

Inputs:
    labels.json       COCO ground truth (bbox = [x, y, w, h])
    predictions.json  [{"file_name": ..., "prediction": [{"class_id", "class_name",
                        "score", "bbox": [x1, y1, x2, y2]}, ...]}, ...]

--evaluate-planted (test sets only, off by default): planted test errors are read from
labels.json itself: a "<name>_wrong.jpg" / "<name>_missing.jpg" image is a copy of "<name>.jpg"
with some classes changed / boxes removed, and "<name>_duplicate.jpg" is an exact copy. They are
marked in the CSVs and the summary reports how many the audit catches. Without it, file names
mean nothing.

Outputs (in --out-dir):
    audit_scores.csv  one row per image with every score from every stage
    shortlist.csv     the --budget images worth annotating (pass quality, not a duplicate,
                        most distinctive first)
    duplicate_sets.csv  one row per near-duplicate group: the kept representative + the rest
    image_scores.csv  one row per image, annotation scores only, sorted worst first
    all_box_scores.csv  every box with its raw scores, lowest (most suspicious) first;
                        use this as a review queue or to pick your own cutoff
    box_issues.csv    one row per box flagged by cleanlab's default cutoffs:
                        overlooked -> model sees an object that is NOT annotated (missing label)
                        swap       -> annotated box probably has the wrong class
                        badloc     -> annotated box is poorly placed / sized
    viz/              given labels (left, red) vs predictions (right, blue) for the worst images
    quality_cache.csv, clip_emb.npz  cached measurements, reused on the next run

Usage:
    python cleanlab_data_audit.py
    python cleanlab_data_audit.py --num-viz 100 --out-dir cleanlab_results
    python cleanlab_data_audit.py --skip-quality --skip-duplicates   # annotation audit only
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import matplotlib

matplotlib.use("Agg")  # cleanlab's visualize() calls plt.show(); keep it non-blocking
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFile

from cleanlab.internal.constants import (
    ALPHA,
    BADLOC_THRESHOLD_FACTOR,
    HIGH_PROBABILITY_THRESHOLD,
    LOW_PROBABILITY_THRESHOLD,
    OVERLOOKED_THRESHOLD_FACTOR,
    SWAP_THRESHOLD_FACTOR,
)
from cleanlab.object_detection.filter import (
    _find_label_issues_per_box,
    _get_per_class_ap,
    _process_class_list,
    find_label_issues,
)
from cleanlab.object_detection.rank import (
    _get_valid_inputs_for_compute_scores,
    compute_badloc_box_scores,
    compute_overlooked_box_scores,
    compute_swap_box_scores,
    get_label_quality_scores,
)
from cleanlab.object_detection.summary import visualize

# let truncated / corrupt files raise so we can flag them as low quality
ImageFile.LOAD_TRUNCATED_IMAGES = False


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


def load_data(labels_path, predictions_path, crowd_ignore):
    with open(labels_path) as f:
        coco = json.load(f)
    with open(predictions_path) as f:
        preds_raw = json.load(f)

    # COCO category ids (1..90, with gaps) -> contiguous 0..K-1 for cleanlab
    categories = sorted(coco["categories"], key=lambda c: c["id"])
    cat_to_idx = {c["id"]: i for i, c in enumerate(categories)}
    class_names = {i: c["name"] for i, c in enumerate(categories)}
    K = len(categories)

    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    preds_by_file = {item["file_name"]: item["prediction"] for item in preds_raw}

    images = sorted(coco["images"], key=lambda im: im["file_name"])

    labels, predictions, meta = [], [], []
    dropped_crowd_preds = 0

    for image in images:
        anns = [a for a in anns_by_image[image["id"]] if not a.get("iscrowd", 0)]
        crowd_anns = [a for a in anns_by_image[image["id"]] if a.get("iscrowd", 0)]

        labels.append({
            "bboxes": np.array([xywh_to_xyxy(a["bbox"]) for a in anns], dtype=float).reshape(-1, 4),
            "labels": np.array([cat_to_idx[a["category_id"]] for a in anns], dtype=int),
            "image_name": image["file_name"],
        })

        raw = preds_by_file.get(image["file_name"], [])
        # Crowd regions aren't valid single-object labels, so drop same-class
        # predictions that fall inside them instead of reporting them as missing.
        if raw and crowd_anns and crowd_ignore < 1.0:
            keep = np.ones(len(raw), dtype=bool)
            for cat_id in {a["category_id"] for a in crowd_anns}:
                regions = np.array([xywh_to_xyxy(a["bbox"]) for a in crowd_anns
                                    if a["category_id"] == cat_id], dtype=float)
                idx = [n for n, p in enumerate(raw) if int(p["class_id"]) == cat_id]
                if idx:
                    boxes = np.array([raw[n]["bbox"] for n in idx], dtype=float)
                    keep[np.array(idx)[inside_fraction(boxes, regions) >= crowd_ignore]] = False
            dropped_crowd_preds += int((~keep).sum())
            raw = [p for p, k in zip(raw, keep) if k]

        # cleanlab format: predictions[i][k] = (M_k, 5) array of [x1, y1, x2, y2, score]
        per_class = [[] for _ in range(K)]
        for p in raw:
            per_class[cat_to_idx[int(p["class_id"])]].append([*p["bbox"], p["score"]])
        pred = np.empty(K, dtype=object)
        for k in range(K):
            pred[k] = np.array(per_class[k], dtype=float).reshape(-1, 5)
        predictions.append(pred)

        meta.append({"image": image, "anns": anns})

    print(f"Images: {len(images)} | classes: {K} | "
          f"annotations (non-crowd): {sum(len(l['labels']) for l in labels)} | "
          f"predictions: {sum(sum(len(c) for c in p) for p in predictions)} "
          f"(dropped {dropped_crowd_preds} inside crowd regions)")

    return labels, predictions, meta, class_names


# ============================================================
# A. Image quality
# ============================================================

CV_ISSUES = ["dark", "light", "blurry", "low_information", "odd_aspect_ratio", "odd_size", "grayscale"]
OWN_ISSUES = ["corrupt", "tiny"]  # the only checks CleanVision doesn't cover


def is_corrupt(path):
    try:
        with Image.open(path) as im:
            im.load()  # full decode: truncated files raise here
        return False
    except Exception:
        return True


def audit_quality(meta, images_dir, cache_dir, min_side, reject_issues, workers):
    """One dict per image: every issue flag, every CleanVision score, and the keep/drop gate."""
    print("\n=== A. QUALITY (CleanVision) ===")
    names = [m["image"]["file_name"] for m in meta]
    cache = os.path.join(cache_dir, "quality_cache.csv")

    df = None
    if os.path.isfile(cache):
        df = pd.read_csv(cache, index_col="file_name")
        if set(names) <= set(df.index):
            print(f"Reusing {cache}")
            df = df.loc[names]
        else:
            df = None

    if df is None:
        paths = [os.path.normpath(os.path.join(images_dir, n)) for n in names]
        print(f"Checking {len(paths)} images are readable...")
        with ThreadPoolExecutor(workers) as ex:
            corrupt = list(ex.map(is_corrupt, paths))

        from cleanvision import Imagelab  # imported here so --skip-quality doesn't need it
        readable = [p for p, c in zip(paths, corrupt) if not c]
        print(f"Running CleanVision on {len(readable)} readable images...")
        lab = Imagelab(filepaths=readable, verbose=False)
        lab.find_issues({k: {} for k in CV_ISSUES}, verbose=False)
        path_to_name = dict(zip(paths, names))
        df = lab.issues.copy()
        df.index = [path_to_name[os.path.normpath(p)] for p in df.index]
        df = df.reindex(names)  # corrupt images -> NaN rows
        for k in CV_ISSUES:
            df[f"is_{k}_issue"] = df[f"is_{k}_issue"].fillna(False).astype(bool)
        df.insert(0, "corrupt", corrupt)
        df.index.name = "file_name"
        df.to_csv(cache)

    min_sides = np.array([min(m["image"].get("width") or 0, m["image"].get("height") or 0) for m in meta])
    corrupt = df["corrupt"].to_numpy(bool)
    flags = {"corrupt": corrupt, "tiny": ~corrupt & (min_sides < min_side)}
    for k in CV_ISSUES:
        flags[k] = df[f"is_{k}_issue"].to_numpy(bool)
    all_issues = OWN_ISSUES + CV_ISSUES

    # `issues` lists EVERY issue an image has; `quality` names only the issues that
    # reject it; `low_quality` is the keep/drop gate.
    rows = []
    for i in range(len(names)):
        issues = [k for k in all_issues if flags[k][i]]
        rejecting = [k for k in issues if k in reject_issues]
        row = {"min_side": int(min_sides[i])}
        for k in all_issues:
            row[k] = bool(flags[k][i])
        for k in CV_ISSUES:
            s = df[f"{k}_score"].iloc[i]
            row[f"{k}_score"] = "" if pd.isna(s) else float(s)
        row.update(issues=";".join(issues), n_issues=len(issues),
                   quality="+".join(rejecting) or "ok", low_quality=bool(rejecting))
        rows.append(row)

    # read-out to help tune --min-side / --reject-issues
    print(f"{'issue':20s} {'images':>7s}  rejects?")
    for k in all_issues:
        print(f"{k:20s} {flags[k].sum():7d}  {'yes' if k in reject_issues else 'no'}")
    print("score p5 / p50 / p95 (lower = worse):")
    for k in CV_ISSUES:
        v = df[f"{k}_score"].dropna().to_numpy(float)
        if len(v):
            p5, p50, p95 = np.percentile(v, [5, 50, 95])
            print(f"  {k:18s} {p5:8.3f} {p50:8.3f} {p95:8.3f}")
    print(f"low_quality total: {sum(r['low_quality'] for r in rows)} / {len(rows)}")
    return rows


# ============================================================
# B. Duplicates / near-duplicates
# ============================================================

def compute_clip_embeddings(paths, cache_path, weights, device, batch_size, workers):
    """L2-normalised CLIP ViT-B/32 embeddings, plus a mask of images that could be read.

    weights: local OpenAI CLIP-ViT-B-32.pt (e.g. FiftyOne's zoo copy), or None to download.
    """
    if os.path.isfile(cache_path):
        z = np.load(cache_path)
        if list(z["paths"]) == list(paths):
            print(f"Reusing {cache_path}")
            return z["emb"], z["ok"]

    import open_clip  # imported here so --skip-duplicates doesn't need them
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if weights and os.path.isfile(weights):
        model = open_clip.load_openai_model(weights, precision="fp32", device=device)
        preprocess = open_clip.image_transform(model.visual.image_size, is_train=False)
    else:
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device=device)
    model.eval()

    def load(path):
        try:
            with Image.open(path) as im:
                return preprocess(im.convert("RGB"))
        except Exception:
            return None

    emb = np.zeros((len(paths), model.visual.output_dim), dtype=np.float32)
    ok = np.zeros(len(paths), dtype=bool)
    print(f"Computing CLIP embeddings for {len(paths)} images (device={device})...")
    with ThreadPoolExecutor(workers) as ex, torch.no_grad():
        for start in range(0, len(paths), batch_size):
            idx = list(range(start, min(start + batch_size, len(paths))))
            tensors = list(ex.map(load, [paths[i] for i in idx]))
            idx = [i for i, t in zip(idx, tensors) if t is not None]
            if not idx:
                continue
            x = torch.stack([t for t in tensors if t is not None]).to(device)
            f = model.encode_image(x)
            emb[idx] = (f / f.norm(dim=-1, keepdim=True)).float().cpu().numpy()
            ok[idx] = True
            if (start // batch_size) % 20 == 0:
                print(f"  {start + len(idx)} / {len(paths)}")

    np.savez(cache_path, paths=np.array(paths), emb=emb, ok=ok)
    return emb, ok


def duplicate_groups(near_duplicate_sets):
    """Datalab gives each item's neighbours within the radius; merge them into groups."""
    parent = list(range(len(near_duplicate_sets)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, neighbours in enumerate(near_duplicate_sets):
        for j in neighbours:
            ri, rj = find(i), find(int(j))
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)

    groups = defaultdict(list)
    for i in range(len(parent)):
        groups[find(i)].append(i)
    return [sorted(g) for g in groups.values() if len(g) > 1]


def audit_duplicates(meta, images_dir, cache_dir, dup_thresh, weights, device, batch_size, workers):
    """One dict per image (duplicate / outlier / uniqueness scores) and the duplicate groups."""
    print("\n=== B. DUPLICATES (CLIP + Datalab) ===")
    from cleanlab import Datalab
    from sklearn.neighbors import NearestNeighbors

    names = [m["image"]["file_name"] for m in meta]
    paths = [os.path.normpath(os.path.join(images_dir, n)) for n in names]
    emb, ok = compute_clip_embeddings(paths, os.path.join(cache_dir, "clip_emb.npz"),
                                      weights, device, batch_size, workers)
    ok_idx = np.where(ok)[0]  # unreadable images get no duplicate scores
    feats = emb[ok_idx]

    lab = Datalab(data={"file_name": [names[i] for i in ok_idx]}, verbosity=0)
    lab.find_issues(
        features=feats,
        issue_types={"near_duplicate": {"metric": "cosine", "threshold": dup_thresh}, "outlier": {}},
    )
    issues = lab.get_issues()

    # Keep the first image (by file name) of each group; the rest are redundant.
    groups = [[int(ok_idx[j]) for j in g]
              for g in duplicate_groups(lab.get_info("near_duplicate")["near_duplicate_sets"])]
    is_dup = np.zeros(len(names), dtype=bool)
    group_of = {}
    for gid, g in enumerate(groups):
        for i in g:
            group_of[i] = gid
        is_dup[g[1:]] = True

    # nearest-neighbour cosine distance (small == likely duplicate) and uniqueness
    # (mean distance to the 3 nearest others, scaled 0..1, high == distinctive)
    nn = NearestNeighbors(n_neighbors=min(4, len(feats)), metric="cosine").fit(feats)
    dist, _ = nn.kneighbors(feats)  # col 0 == self
    uniq = dist[:, 1:].mean(axis=1)
    uniq = (uniq - uniq.min()) / (uniq.max() - uniq.min() + 1e-12)

    rows = [{"is_duplicate": False, "dup_group": "", "near_duplicate_score": "",
             "is_outlier": "", "outlier_score": "", "nn_distance": "", "uniqueness": ""}
            for _ in names]
    for j, i in enumerate(ok_idx):
        rows[i].update(
            is_duplicate=bool(is_dup[i]), dup_group=group_of.get(i, ""),
            near_duplicate_score=float(issues["near_duplicate_score"].iloc[j]),
            is_outlier=bool(issues["is_outlier_issue"].iloc[j]),
            outlier_score=float(issues["outlier_score"].iloc[j]),
            nn_distance=float(dist[j, 1]), uniqueness=float(uniq[j]),
        )

    print(f"Near-duplicate groups at threshold={dup_thresh} x median NN distance: {len(groups)} "
          f"({int(is_dup.sum())} redundant images)")
    print(f"Outliers: {int(issues['is_outlier_issue'].sum())}")
    return rows, groups


# ============================================================
# C. Per-box issues
# ============================================================

def flatten_prediction(pred):
    """Same ordering cleanlab uses internally: grouped by class index."""
    boxes, classes, scores = [], [], []
    for k, arr in enumerate(pred):
        for row in arr:
            boxes.append(row[:4])
            classes.append(k)
            scores.append(row[4])
    return np.array(boxes, dtype=float).reshape(-1, 4), np.array(classes, dtype=int), np.array(scores)


def compute_box_issues(labels, predictions, swap_prob_threshold=HIGH_PROBABILITY_THRESHOLD,
                       overlooked_prob_threshold=HIGH_PROBABILITY_THRESHOLD):
    """
    Per-box scores (lower = more likely an error) and flags, using the same
    per-class thresholds cleanlab.object_detection.filter.find_label_issues uses.

    swap_prob_threshold: a different-class prediction only counts as evidence of a
    wrong class above this confidence (cleanlab default 0.95).
    overlooked_prob_threshold: only predictions above this confidence are scored as
    possibly-unannotated objects (cleanlab default 0.95).

    NOTE: relies on a few private cleanlab helpers (tested with cleanlab 2.9).
    """
    aux = _get_valid_inputs_for_compute_scores(ALPHA, labels, predictions)

    overlooked = compute_overlooked_box_scores(
        alpha=ALPHA, high_probability_threshold=overlooked_prob_threshold, auxiliary_inputs=aux)
    badloc = compute_badloc_box_scores(
        alpha=ALPHA, low_probability_threshold=LOW_PROBABILITY_THRESHOLD, auxiliary_inputs=aux)
    swap = compute_swap_box_scores(
        alpha=ALPHA, high_probability_threshold=swap_prob_threshold, auxiliary_inputs=aux)

    per_class_ap = _get_per_class_ap(labels, predictions)
    lab_thresh = _process_class_list([l["labels"] for l in labels], per_class_ap)
    pred_thresh = _process_class_list([flatten_prediction(p)[1] for p in predictions], per_class_ap)

    # _find_label_issues_per_box overwrites NaNs in place, so pass copies
    flags = {
        "overlooked": _find_label_issues_per_box(
            [s.copy() for s in overlooked], pred_thresh, OVERLOOKED_THRESHOLD_FACTOR),
        "badloc": _find_label_issues_per_box(
            [s.copy() for s in badloc], lab_thresh, BADLOC_THRESHOLD_FACTOR),
        "swap": _find_label_issues_per_box(
            [s.copy() for s in swap], lab_thresh, SWAP_THRESHOLD_FACTOR),
    }
    scores = {"overlooked": overlooked, "badloc": badloc, "swap": swap}
    return scores, flags


def match_counts(label, pboxes, pclasses, pscores, iou_thresh, conf_thresh):
    """COCO-style greedy match (highest confidence first, same class) -> (tp, fp, fn)."""
    keep = pscores >= conf_thresh
    pb, pc, ps = pboxes[keep], pclasses[keep], pscores[keep]
    ious = iou_matrix(pb, label["bboxes"])
    matched = np.zeros(len(label["labels"]), dtype=bool)
    tp = 0
    for p in np.argsort(-ps):
        cand = np.where((label["labels"] == pc[p]) & ~matched)[0]
        if len(cand):
            g = cand[np.argmax(ious[p, cand])]
            if ious[p, g] >= iou_thresh:
                matched[g] = True
                tp += 1
    return tp, len(pb) - tp, len(matched) - tp


# ============================================================
# Planted test errors (answer key derived from labels.json)
# ============================================================

def find_planted(labels_path):
    """
    {file_name: "wrong" | "duplicate"} for the planted images, and the ids of the
    planted wrong-class annotations: boxes in a *_wrong image whose class differs
    from the same box in the original image. None if nothing was planted.
    """
    with open(labels_path) as f:
        coco = json.load(f)
    by_name = {im["file_name"]: im for im in coco["images"]}
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

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
    if not planted_images:
        return None
    kinds = list(planted_images.values())
    print(f"Planted in {labels_path}: {kinds.count('wrong')} _wrong images "
          f"({len(planted_anns)} wrong-class boxes), {kinds.count('missing')} _missing images (dropped "
          f"annotations), {kinds.count('duplicate')} _duplicate images")
    return planted_images, planted_anns


# ============================================================
# Main
# ============================================================

def fmt_bbox_xywh(b):
    x1, y1, x2, y2 = b
    return f"[{x1:.1f}, {y1:.1f}, {x2 - x1:.1f}, {y2 - y1:.1f}]"


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["file_name"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", default="labels.json")
    parser.add_argument("--predictions", default="predictions.json")
    parser.add_argument("--images-dir", default="data")
    parser.add_argument("--out-dir", default="cleanlab_results")
    parser.add_argument("--cache-dir", default=None, help="where cached measurements live (default: --out-dir)")
    parser.add_argument("--num-viz", type=int, default=50, help="visualize this many worst images (0 = none)")
    parser.add_argument("--crowd-ignore", type=float, default=0.5,
                        help="drop predictions whose area is >= this fraction inside a same-class iscrowd region "
                             "(prevents false 'missing annotation' flags); 1.0 disables")
    parser.add_argument("--swap-prob-threshold", type=float, default=0.5,
                        help="min confidence for a different-class prediction to count as evidence of a "
                             "wrong class (cleanlab default 0.95 misses most errors on this data)")
    parser.add_argument("--overlooked-prob-threshold", type=float, default=0.5,
                        help="min confidence for a prediction to be scored as a possibly-missing annotation "
                             "(cleanlab default 0.95 leaves almost every prediction unscored)")
    parser.add_argument("--eval-iou", type=float, default=0.5, help="IoU for the TP/FP/FN match counts")
    parser.add_argument("--eval-conf", type=float, default=0.5,
                        help="ignore predictions below this confidence in the TP/FP/FN match counts")
    # A. quality
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--min-side", type=int, default=100, help="px on the short edge; smaller is flagged 'tiny'")
    parser.add_argument("--reject-issues", default="corrupt,tiny,blurry,dark,low_information",
                        help="comma-separated issues that make an image NOT worth annotating (-> low_quality); "
                             f"choices: {','.join(OWN_ISSUES + CV_ISSUES)}")
    # B. duplicates
    parser.add_argument("--skip-duplicates", action="store_true")
    parser.add_argument("--dup-thresh", type=float, default=0.13,
                        help="Datalab near_duplicate threshold, as a fraction of the median nearest-neighbour "
                             "distance (NOT an absolute cosine distance like FiftyOne's)")
    parser.add_argument("--budget", type=int, default=1000, help="how many images you can afford to annotate")
    parser.add_argument("--clip-weights", default=os.path.expanduser("~/fiftyone/__models__/CLIP-ViT-B-32.pt"),
                        help="local OpenAI CLIP ViT-B/32 checkpoint (default: FiftyOne's zoo copy); "
                             "downloaded from Hugging Face if the file doesn't exist")
    parser.add_argument("--device", default=None, help="torch device for CLIP (default: cuda if available)")
    parser.add_argument("--clip-batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8, help="threads for image decoding")
    parser.add_argument("--evaluate-planted", action="store_true",
                        help="test sets only: mark and report planted *_wrong / *_missing / *_duplicate "
                             "images (default: off, file names mean nothing)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cache_dir = args.cache_dir or args.out_dir
    os.makedirs(cache_dir, exist_ok=True)
    reject_issues = {s.strip() for s in args.reject_issues.split(",") if s.strip()}

    labels, predictions, meta, class_names = load_data(args.labels, args.predictions, args.crowd_ignore)

    quality_rows = None if args.skip_quality else audit_quality(
        meta, args.images_dir, cache_dir, args.min_side, reject_issues, args.workers)
    dup_rows, dup_groups = (None, []) if args.skip_duplicates else audit_duplicates(
        meta, args.images_dir, cache_dir, args.dup_thresh, args.clip_weights, args.device, args.clip_batch,
        args.workers)

    print("\n=== C. ANNOTATIONS (ObjectLab) ===")
    print("Scoring images...")
    quality = get_label_quality_scores(labels, predictions)
    issue_idx = find_label_issues(labels, predictions, return_indices_ranked_by_score=True)
    is_issue = np.zeros(len(labels), dtype=bool)
    is_issue[issue_idx] = True

    print("Scoring individual boxes...")
    box_scores, box_flags = compute_box_issues(labels, predictions, args.swap_prob_threshold,
                                               args.overlooked_prob_threshold)

    # --------------------------------------------------------
    # box_issues.csv
    # --------------------------------------------------------
    box_rows = []
    all_rows = []  # every box with its raw scores, for ranking / picking your own cutoff
    counts = defaultdict(lambda: defaultdict(int))
    match = []  # (tp, fp, fn) per image

    for i, (label, pred, m) in enumerate(zip(labels, predictions, meta)):
        image = m["image"]
        pboxes, pclasses, pscores = flatten_prediction(pred)
        ious = iou_matrix(label["bboxes"], pboxes)
        match.append(match_counts(label, pboxes, pclasses, pscores, args.eval_iou, args.eval_conf))
        counts[i]["n_pred"] = len(pboxes)

        # Every annotated box: swap (wrong class) and badloc (bad box) scores
        for j in range(len(label["labels"])):
            swap_s = np.nan_to_num(box_scores["swap"][i][j], nan=1.0)
            badloc_s = np.nan_to_num(box_scores["badloc"][i][j], nan=1.0)
            if len(pboxes):
                best = int(np.argmax(ious[j]))
                sugg, conf, iou = class_names[pclasses[best]], pscores[best], ious[j, best]
            else:
                sugg, conf, iou = "", "", ""
            all_rows.append({
                "file_name": image["file_name"], "image_id": image["id"], "box_source": "annotation",
                "annotation_id": m["anns"][j]["id"], "given_class": class_names[label["labels"][j]],
                "swap_score": swap_s, "badloc_score": badloc_s, "overlooked_score": "",
                "min_score": min(swap_s, badloc_s),
                "closest_pred_class": sugg, "closest_pred_conf": conf, "closest_pred_iou": iou,
                "bbox_xywh": fmt_bbox_xywh(label["bboxes"][j]),
                "flagged": bool(box_flags["swap"][i][j] or box_flags["badloc"][i][j]),
            })

        # Predicted boxes cleanlab considers possibly-missing annotations
        for j in range(len(pboxes)):
            ov = box_scores["overlooked"][i][j]
            if np.isnan(ov):
                continue  # low confidence or overlaps a label: not an overlooked candidate
            all_rows.append({
                "file_name": image["file_name"], "image_id": image["id"], "box_source": "prediction",
                "annotation_id": "", "given_class": "",
                "swap_score": "", "badloc_score": "", "overlooked_score": ov, "min_score": ov,
                "closest_pred_class": class_names[pclasses[j]], "closest_pred_conf": pscores[j],
                "closest_pred_iou": "", "bbox_xywh": fmt_bbox_xywh(pboxes[j]),
                "flagged": bool(box_flags["overlooked"][i][j]),
            })

        for j in np.where(box_flags["overlooked"][i])[0]:
            if len(pboxes) == 0:
                continue  # placeholder flag for images with no predictions
            box_rows.append({
                "file_name": image["file_name"], "image_id": image["id"],
                "issue_type": "overlooked", "issue_score": box_scores["overlooked"][i][j],
                "annotation_id": "", "given_class": "",
                "suggested_class": class_names[pclasses[j]], "pred_confidence": pscores[j],
                "bbox_xywh": fmt_bbox_xywh(pboxes[j]), "iou_with_match": "",
            })
            counts[i]["overlooked"] += 1

        for issue_type in ("swap", "badloc"):
            for j in np.where(box_flags[issue_type][i])[0]:
                if len(label["labels"]) == 0:
                    continue
                # closest prediction for context (what the model thinks is there)
                if len(pboxes):
                    if issue_type == "badloc":
                        same = np.where(pclasses == label["labels"][j])[0]
                        cand = same if len(same) else np.arange(len(pboxes))
                    else:
                        cand = np.arange(len(pboxes))
                    best = cand[np.argmax(ious[j, cand])]
                    sugg, conf, iou = class_names[pclasses[best]], pscores[best], ious[j, best]
                else:
                    sugg, conf, iou = "", "", ""
                box_rows.append({
                    "file_name": image["file_name"], "image_id": image["id"],
                    "issue_type": issue_type, "issue_score": box_scores[issue_type][i][j],
                    "annotation_id": m["anns"][j]["id"], "given_class": class_names[label["labels"][j]],
                    "suggested_class": sugg, "pred_confidence": conf,
                    "bbox_xywh": fmt_bbox_xywh(label["bboxes"][j]), "iou_with_match": iou,
                })
                counts[i][issue_type] += 1

    # --------------------------------------------------------
    # Planted test errors: mark them so you can see whether they're caught
    # --------------------------------------------------------
    planted = find_planted(args.labels) if args.evaluate_planted else None
    if planted:
        planted_images, planted_anns = planted
        for r in box_rows + all_rows:
            r["is_planted_error"] = r["annotation_id"] in planted_anns if r["annotation_id"] != "" else ""

    box_rows.sort(key=lambda r: r["issue_score"])
    box_csv = os.path.join(args.out_dir, "box_issues.csv")
    write_csv(box_csv, box_rows)

    # Annotations by swap score (wrong class), predictions by overlooked score
    # (missing label); min_score would bury wrong classes under badloc noise.
    all_rows.sort(key=lambda r: r["swap_score"] if r["box_source"] == "annotation" else r["overlooked_score"])
    all_csv = os.path.join(args.out_dir, "all_box_scores.csv")
    write_csv(all_csv, all_rows)

    # --------------------------------------------------------
    # image_scores.csv
    # --------------------------------------------------------
    order = np.argsort(quality)
    image_csv = os.path.join(args.out_dir, "image_scores.csv")
    with open(image_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "image_id", "label_quality_score", "has_issue",
                         "n_overlooked", "n_swap", "n_badloc", "n_annotations"])
        for i in order:
            image = meta[i]["image"]
            writer.writerow([image["file_name"], image["id"], f"{quality[i]:.4f}", bool(is_issue[i]),
                             counts[i]["overlooked"], counts[i]["swap"], counts[i]["badloc"],
                             len(labels[i]["labels"])])

    # --------------------------------------------------------
    # audit_scores.csv: one row per image, every stage that ran
    # --------------------------------------------------------
    audit_rows = []
    for i, m in enumerate(meta):
        image = m["image"]
        row = {"file_name": image["file_name"], "image_id": image["id"],
               "width": image.get("width", ""), "height": image.get("height", "")}
        if quality_rows:
            row.update(quality_rows[i])
        if dup_rows:
            row.update(dup_rows[i])
        tp, fp, fn = match[i]
        row.update({
            "label_quality_score": round(float(quality[i]), 4), "has_issue": bool(is_issue[i]),
            "n_overlooked": counts[i]["overlooked"], "n_swap": counts[i]["swap"], "n_badloc": counts[i]["badloc"],
            "eval_tp": tp, "eval_fp": fp, "eval_fn": fn,
            "n_annotations": len(labels[i]["labels"]), "n_pred": counts[i]["n_pred"],
        })
        if planted:
            row["planted"] = planted_images.get(image["file_name"], "")
        audit_rows.append(row)
    audit_csv = os.path.join(args.out_dir, "audit_scores.csv")
    write_csv(audit_csv, audit_rows)

    if dup_rows:
        dup_csv = os.path.join(args.out_dir, "duplicate_sets.csv")
        write_csv(dup_csv, [{"dup_group": gid, "keep": meta[g[0]]["image"]["file_name"],
                             "n_redundant": len(g) - 1,
                             "redundant": ";".join(meta[i]["image"]["file_name"] for i in g[1:])}
                            for gid, g in enumerate(dup_groups)])

    # --------------------------------------------------------
    # shortlist.csv: passes the quality gate, not redundant, most distinctive first
    # --------------------------------------------------------
    shortlist = []
    if dup_rows:
        candidates = [r for r in audit_rows
                      if not r.get("low_quality", False) and not r["is_duplicate"] and r["uniqueness"] != ""]
        candidates.sort(key=lambda r: r["uniqueness"], reverse=True)
        shortlist = [{"rank": n, "file_name": r["file_name"], "image_id": r["image_id"],
                      "uniqueness": r["uniqueness"], "label_quality_score": r["label_quality_score"],
                      "has_issue": r["has_issue"]}
                     for n, r in enumerate(candidates[:args.budget], start=1)]
        shortlist_csv = os.path.join(args.out_dir, "shortlist.csv")
        write_csv(shortlist_csv, shortlist)

    # --------------------------------------------------------
    # Visualizations
    # --------------------------------------------------------
    if args.num_viz > 0:
        viz_dir = os.path.join(args.out_dir, "viz")
        os.makedirs(viz_dir, exist_ok=True)
        print(f"Saving {min(args.num_viz, len(issue_idx))} visualizations to {viz_dir} ...")
        for rank, i in enumerate(issue_idx[:args.num_viz], start=1):
            name = meta[i]["image"]["file_name"]
            c = counts[i]
            visualize(
                os.path.join(args.images_dir, name),
                label=labels[i],
                prediction=predictions[i],
                class_names=class_names,
                overlay=False,
                save_path=os.path.join(
                    viz_dir,
                    f"{rank:04d}_{os.path.splitext(name)[0]}"
                    f"_o{c['overlooked']}_s{c['swap']}_b{c['badloc']}.png"),
            )
            plt.close("all")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    by_type = defaultdict(int)
    for r in box_rows:
        by_type[r["issue_type"]] += 1

    print()
    print("=" * 60)
    if quality_rows:
        print(f"Low-quality images (rejected):     {sum(r['low_quality'] for r in quality_rows)} / {len(labels)}")
    if dup_rows:
        print(f"Redundant near-duplicates:         {sum(r['is_duplicate'] for r in dup_rows)} "
              f"in {len(dup_groups)} groups")
        print(f"Shortlist (budget {args.budget}):           {len(shortlist)} images")
    print(f"Images flagged with label issues:  {len(issue_idx)} / {len(labels)}")
    print(f"  overlooked (missing annotation): {by_type['overlooked']} boxes")
    print(f"  swap       (wrong class):        {by_type['swap']} boxes")
    print(f"  badloc     (bad box placement):  {by_type['badloc']} boxes")
    print(f"Match counts (IoU>={args.eval_iou}, conf>={args.eval_conf}): "
          f"TP={sum(t for t, _, _ in match)} FP={sum(f for _, f, _ in match)} FN={sum(n for _, _, n in match)}")
    if planted:
        ann_rows = [r for r in all_rows if r["box_source"] == "annotation"]  # sorted by swap score
        scored = [r for r in ann_rows if r["is_planted_error"] is True]
        print(f"Planted wrong-class boxes: {len(scored)} scored "
              f"({len(planted_anns) - len(scored)} crowd, never scored)")
        print(f"  flagged (swap or badloc):        {sum(r['flagged'] for r in scored)} / {len(scored)}")
        for k in (500, 1000, 2000, 5000):
            hit = sum(r["is_planted_error"] is True for r in ann_rows[:k])
            print(f"  top {k:5d} by swap score: {hit:4d} planted ({hit / max(len(scored), 1):4.0%} recall, "
                  f"{hit / k:4.0%} precision)")
        for kind, caught_by in (("wrong", "with a label issue"), ("missing", "with a label issue"),
                                ("duplicate", "flagged is_duplicate")):
            idx = [i for i, m in enumerate(meta) if planted_images.get(m["image"]["file_name"]) == kind]
            if not idx:
                continue
            if kind != "duplicate":
                caught = int(is_issue[idx].sum())
            elif dup_rows:
                caught = sum(dup_rows[i]["is_duplicate"] for i in idx)
            else:
                continue
            print(f"  _{kind} images {caught_by}: {caught} / {len(idx)}")
    print("=" * 60)
    print("Worst 10 images:")
    for i in issue_idx[:10]:
        c = counts[i]
        print(f"  {meta[i]['image']['file_name']}  score={quality[i]:.3f}  "
              f"overlooked={c['overlooked']} swap={c['swap']} badloc={c['badloc']}")
    print()
    print(f"Wrote {audit_csv}  (one row per image, every stage)")
    if shortlist:
        print(f"Wrote {shortlist_csv}")
    if dup_rows:
        print(f"Wrote {dup_csv}")
    print(f"Wrote {image_csv}")
    print(f"Wrote {box_csv}")
    print(f"Wrote {all_csv}  (every box, sorted by score: review from the top)")


if __name__ == "__main__":
    main()
