"""
Black box: FiftyOne Brain mistakenness for one model's predictions.

Loads the predictions file into the persistent FiftyOne dataset (field pred_<tag>), runs
fob.compute_mistakenness against the ground truth, and writes the black-box contract
(see blackbox_common.py) to --out-dir:

    s_mistakenness       max over the image's annotations of how likely the label is wrong (0-1)
    s_possible_missing   predictions FiftyOne thinks are unannotated objects (confidence > --missing-conf)
    s_possible_spurious  annotations FiftyOne thinks aren't there
    flagged              mistakenness >= --flag-mistakenness, or any possible missing / spurious
    reason               e.g. "annotated pizza: mistakenness 0.99; possible missing: car 0.98"

--predictions zoo uses the `predictions` field fiftyone_data_audit.py already computed (the
FiftyOne zoo Faster R-CNN: same weights as predictions_rcnn.json, so it's in group "rcnn").
Adds pred_<tag> / mistakenness_<tag> / possible_missing_<tag> / possible_spurious_<tag>
fields to the dataset; the App doesn't need to be running.

Usage:
    python blackbox_fiftyone.py                                      # predictions_rcnn.json
    python blackbox_fiftyone.py --predictions predictions_yolo.json
    python blackbox_fiftyone.py --predictions zoo --reuse
"""

import argparse
import json
import os

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.brain.internal.core.mistakenness as fbm

# import fiftyone_data_audit as fda  # dataset config + loaders (importing it doesn't launch the App)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")) 
from blackbox_common import SCORES_ROOT, image_paths, predictions_tag, write_blackbox

class FDA:
    def __init__(self):
        self.DATASET_NAME = "data-audit"
        self.GT_FIELD   = "ground_truth"   # where COCO boxes land
        self.PRED_FIELD = "predictions"    # detector's independent 2nd opinion

fda = FDA()

def load_predictions(dataset, predictions_path, pred_field, min_score):
    """Write a predictions_*.json file (absolute xyxy boxes) into pred_field as fo.Detections."""
    with open(predictions_path) as f:
        preds_by_file = {item["file_name"]: item["prediction"] for item in json.load(f)}
    if dataset.count("metadata") < len(dataset):
        dataset.compute_metadata()
    detections, n = [], 0
    for fp, w, h in zip(*dataset.values(["filepath", "metadata.width", "metadata.height"])):
        dets = [fo.Detection(label=p["class_name"], confidence=p["score"],
                             bounding_box=[x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h])
                for p in preds_by_file.get(os.path.basename(fp), []) if p["score"] >= min_score
                for x1, y1, x2, y2 in [p["bbox"]]]
        n += len(dets)
        detections.append(fo.Detections(detections=dets))
    dataset.set_values(pred_field, detections)
    print(f"Loaded {n} predictions >= {min_score} into '{pred_field}'")


def add_new_images(full, labels_path, images_dir, gt_field):
    """Import --labels' images the dataset doesn't have yet (e.g. new _missing copies), the same
    way fiftyone_data_audit.py imported the rest (COCO importer, boxes only)."""
    with open(labels_path) as f:
        coco = json.load(f)
    have = {os.path.basename(fp) for fp in full.values("filepath")}
    new = [im for im in coco["images"] if im["file_name"] not in have]
    if not new:
        return
    ids = {im["id"] for im in new}
    tmp = os.path.join(os.path.dirname(os.path.abspath(labels_path)), "_new_images_tmp.json")
    with open(tmp, "w") as f:
        json.dump({**coco, "images": new, "annotations": [a for a in coco["annotations"] if a["image_id"] in ids]}, f)
    try:
        full.add_dir(dataset_type=fo.types.COCODetectionDataset, data_path=images_dir, labels_path=tmp,
                     label_field=gt_field, label_types="detections")  # a str: no "_detections" suffix
    finally:
        os.remove(tmp)
    full.compute_metadata()
    print(f"Added {len(new)} new images from {labels_path} to dataset '{full.name}'")


def sync_labels(dataset, labels_path, gt_field):
    """Make the dataset's ground-truth classes match --labels (e.g. after plant_wrong.py rewrote
    labels.json). Detections are matched to annotations by box; only their label changes."""
    with open(labels_path) as f:
        coco = json.load(f)
    cat = {c["id"]: c["name"] for c in coco["categories"]}
    size = {im["file_name"]: (im["width"], im["height"]) for im in coco["images"]}
    by_id = {im["id"]: im["file_name"] for im in coco["images"]}
    want = {}
    for a in coco["annotations"]:
        n = by_id[a["image_id"]]
        w, h = size[n]
        x, y, bw, bh = a["bbox"]
        want.setdefault(n, {})[tuple(round(v, 3) for v in (x / w, y / h, bw / w, bh / h))] = cat[a["category_id"]]
    key = lambda d: tuple(round(v, 3) for v in d.bounding_box)
    changed = unmatched = 0
    for sample in dataset.select_fields(gt_field):
        dets = sample[gt_field].detections if sample[gt_field] else []
        target = want.get(os.path.basename(sample.filepath), {})
        edits = [(d, target[key(d)]) for d in dets if key(d) in target and d.label != target[key(d)]]
        unmatched += sum(key(d) not in target for d in dets)
        if edits:
            for d, label in edits:
                d.label = label
            sample.save()
            changed += 1
    if changed or unmatched:
        print(f"Synced ground truth with {labels_path}: relabelled boxes in {changed} samples"
              + (f"; {unmatched} boxes have no match in {labels_path} (left as they are)" if unmatched else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", default="labels.json", help="only used for the image list and paths")
    parser.add_argument("--predictions", default="predictions_rcnn.json",
                        help="predictions_*.json, or 'zoo' for the dataset's existing `predictions` field")
    parser.add_argument("--images-dir", default="data")
    parser.add_argument("--dataset", "--audit-name", dest="dataset", default=fda.DATASET_NAME,
                        help="persistent FiftyOne dataset; created from --labels if it doesn't exist. "
                             "Use one per image collection: images are matched by file name "
                             "(--audit-name: what run_audit.py passes)")
    parser.add_argument("--min-score", type=float, default=0.01, help="ignore predictions below this score")
    parser.add_argument("--flag-mistakenness", type=float, default=0.5)
    parser.add_argument("--missing-conf", type=float, default=0.8,
                        help="unmatched predictions above this confidence are possible missing annotations "
                             "(FiftyOne hard-codes 0.95)")
    parser.add_argument("--out-dir", default=None, help=f"black-box output (default: {SCORES_ROOT}/fiftyone_<tag>)")
    parser.add_argument("--reuse", action="store_true", help="don't recompute if the mistakenness field exists")
    args = parser.parse_args()

    zoo = args.predictions == "zoo"
    tag = "rcnn" if zoo else predictions_tag(args.predictions)
    name = "fiftyone_zoo" if zoo else f"fiftyone_{tag}"
    out_dir = args.out_dir or os.path.join(SCORES_ROOT, name)
    suffix = "" if zoo else f"_{tag}"
    pred_field = fda.PRED_FIELD if zoo else f"pred_{tag}"
    m_field, miss_field, spur_field = (f"mistakenness{suffix}", f"possible_missing{suffix}",
                                       f"possible_spurious{suffix}")

    if args.dataset in fo.list_datasets():
        full = fo.load_dataset(args.dataset)
    else:  # a new dataset: built from --labels / --images-dir only (add_new_images below)
        full = fo.Dataset(args.dataset)
        full.persistent = True
        print(f"Created FiftyOne dataset '{args.dataset}'")
    schema = full.get_field_schema()
    gt_field = next((f for f in (f"{fda.GT_FIELD}_detections", fda.GT_FIELD) if f in schema), fda.GT_FIELD)
    add_new_images(full, args.labels, args.images_dir, gt_field)
    schema = full.get_field_schema()
    paths = image_paths(args.labels, args.images_dir)
    ids, fps = full.values(["id", "filepath"])
    dataset = full.select([i for i, fp in zip(ids, fps) if os.path.basename(fp) in paths])  # only --labels' images
    print(f"{len(dataset)} of the dataset's {len(full)} samples are in {args.labels}")
    sync_labels(dataset, args.labels, gt_field)

    if not (args.reuse and m_field in schema and dataset.count(m_field) == len(dataset)):
        if not zoo:
            load_predictions(dataset, args.predictions, pred_field, args.min_score)
        elif fda.PRED_FIELD not in schema:
            raise SystemExit(f"No '{pred_field}' field: run fiftyone_data_audit.py first, or pass a predictions file")
        fbm._MISSED_CONFIDENCE_THRESHOLD = args.missing_conf  # module constant, read at call time
        print(f"compute_mistakenness({pred_field} vs {gt_field})...")
        fob.compute_mistakenness(dataset, pred_field, label_field=gt_field, mistakenness_field=m_field,
                                 missing_field=miss_field, spurious_field=spur_field)

    names = [os.path.basename(fp) for fp in dataset.values("filepath")]
    m, miss, spur = dataset.values([m_field, miss_field, spur_field])
    gt_labels, gt_m = dataset.values([f"{gt_field}.detections.label", f"{gt_field}.detections.{m_field}"])
    p_labels, p_conf, p_miss = dataset.values([f"{pred_field}.detections.label", f"{pred_field}.detections.confidence",
                                               f"{pred_field}.detections.{miss_field}"])

    by_name = {}
    for i, n in enumerate(names):
        parts = []
        scored = [(s, lab) for s, lab in zip(gt_m[i] or [], gt_labels[i] or []) if s is not None]
        if scored:
            s, lab = max(scored)
            if s >= args.flag_mistakenness:
                parts.append(f"annotated {lab}: mistakenness {s:.2f}")
        missing = sorted(((c, lab) for c, lab, mi in zip(p_conf[i] or [], p_labels[i] or [], p_miss[i] or []) if mi),
                         reverse=True)
        if missing:
            parts.append(f"possible missing: {missing[0][1]} {missing[0][0]:.2f}"
                         + (f" (+{len(missing) - 1} more)" if len(missing) > 1 else ""))
        if spur[i]:
            parts.append(f"{spur[i]} possible spurious annotation(s)")
        by_name[n] = {
            "s_mistakenness": round(m[i], 4) if m[i] is not None else 0.0,
            "s_possible_missing": miss[i] or 0,
            "s_possible_spurious": spur[i] or 0,
            "flagged": (m[i] or 0) >= args.flag_mistakenness or bool(miss[i]) or bool(spur[i]),
            "reason": "; ".join(parts),
        }

    rows = [{"file_name": n, "file_path": path, **by_name.get(n, {"flagged": False})}
            for n, path in paths.items()]
    write_blackbox(out_dir, rows, name=name, method="fiftyone", group=tag,
                   signals={"s_mistakenness": "max FiftyOne mistakenness over the image's annotations",
                            "s_possible_missing": "count of likely unannotated objects",
                            "s_possible_spurious": "count of likely nonexistent annotations"},
                   weights={"s_mistakenness": 1, "s_possible_missing": 1, "s_possible_spurious": 0.25},
                   params={"predictions": args.predictions, "dataset": args.dataset, "pred_field": pred_field,
                           "min_score": args.min_score, "flag_mistakenness": args.flag_mistakenness,
                           "missing_conf": args.missing_conf})


if __name__ == "__main__":
    main()
