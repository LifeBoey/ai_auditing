"""
Run the three BCCD-trained detectors over every image in bccdp_images/ and write
COCO-results-format predictions to prediction_jsons/predictions_bccdp_<model>.json.

Expects:
    bccd_training_runs/<model>/...best.pt   (from train_bccd_detectors.py)
    bccdp_ground_truth.json                 (from make_bccdp.py; supplies the image ids)

Each output file has one entry per image (every image, even with no detections), in the
format cv_annotation/flag_annotations.py expects (same as predictions_yolo11x.json):
    [{"image_id", "file_name",
      "prediction": [{"class_id", "class_name", "score", "bbox": [x1, y1, x2, y2]}]}]
class_id is the category id of bccdp_ground_truth.json.

Usage:
    python predict_bccdp.py
    python predict_bccdp.py --models yolo11n --conf 0.25
    python predict_bccdp.py --reformat    # convert existing flat COCO-results files, no inference
    python predict_bccdp.py --dataset bccd  # unperturbed bccd_images -> predictions_bccd_<model>.json
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train_bccd_detectors import (
    MODEL_CONFIGS,
    OUTPUT_DIR,
    CocoDetection,
    build_torchvision_model,
    coco_from_dict,
    coco_metrics,
    collate,
    fmt_metrics,
    predict_torchvision,
    require_gpu,
)


# ============================================================
# CONFIGURATION
# ============================================================

HERE = Path(__file__).resolve().parent
PRED_DIR = HERE / "prediction_jsons"
# --dataset picks the images, the ground truth (for image ids) and the output prefix
DATASETS = ("bccdp", "bccd")
DATASET = "bccdp"
IMAGE_DIR = HERE / f"{DATASET}_images"
GT_JSON = HERE / f"{DATASET}_ground_truth.json"

BATCH_SIZE = 16


def pred_path(name):
    return PRED_DIR / f"predictions_{DATASET}_{name}.json"


def to_per_image(flat, gt, cat_names):
    """Flat COCO results (xywh) -> per-image entries with xyxy boxes, highest score first."""
    by_image = {im["id"]: [] for im in gt["images"]}
    for r in flat:
        x, y, w, h = r["bbox"]
        by_image[r["image_id"]].append({
            "class_id": r["category_id"], "class_name": cat_names[r["category_id"]], "score": r["score"],
            "bbox": [x, y, round(x + w, 2), round(y + h, 2)],
        })
    return [{"image_id": im["id"], "file_name": im["file_name"],
             "prediction": sorted(by_image[im["id"]], key=lambda p: -p["score"])}
            for im in gt["images"]]


def weights_path(name):
    if MODEL_CONFIGS[name]["kind"] == "ultralytics":
        return OUTPUT_DIR / name / "weights" / "best.pt"
    return OUTPUT_DIR / name / "best.pt"


# ============================================================
# PREDICTION
# ============================================================

def predict_yolo(name, gt, cats, conf):
    from ultralytics import YOLO

    model = YOLO(str(weights_path(name)))
    idx_to_cat = {i: c["id"] for i, c in enumerate(cats)}
    imgsz = MODEL_CONFIGS[name]["imgsz"]
    results = []
    images = gt["images"]
    for i in range(0, len(images), BATCH_SIZE):
        batch = images[i:i + BATCH_SIZE]
        outs = model.predict([str(IMAGE_DIR / im["file_name"]) for im in batch], imgsz=imgsz, conf=conf,
                             iou=0.7, max_det=100, device=0, verbose=False)
        for im, r in zip(batch, outs):
            xyxy = r.boxes.xyxy.cpu().tolist()
            for (x1, y1, x2, y2), s, c in zip(xyxy, r.boxes.conf.cpu().tolist(), r.boxes.cls.cpu().tolist()):
                results.append({"image_id": im["id"], "category_id": idx_to_cat[int(c)],
                                "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                                "score": round(s, 5)})
    return results


def predict_tv(name, gt, cats, conf, device, workers):
    model = build_torchvision_model(name, num_classes=len(cats) + 1)
    model.load_state_dict(torch.load(weights_path(name), map_location="cpu"))
    model.to(device)
    loader = DataLoader(CocoDetection(gt, IMAGE_DIR, train=False), batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=workers, collate_fn=collate, pin_memory=True)
    return [r for r in predict_torchvision(model, loader, device) if r["score"] >= conf]


# ============================================================
# MAIN
# ============================================================

def main():
    global DATASET, IMAGE_DIR, GT_JSON
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODEL_CONFIGS), choices=list(MODEL_CONFIGS))
    ap.add_argument("--conf", type=float, default=0.001,
                    help="minimum score to keep (low default so the files are usable for mAP)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--reformat", action="store_true",
                    help="rewrite existing flat-format prediction files in place instead of running the models")
    ap.add_argument("--dataset", choices=DATASETS, default=DATASET,
                    help="bccdp: perturbed copy (default); bccd: the original, unperturbed data")
    args = ap.parse_args()

    DATASET = args.dataset
    IMAGE_DIR = HERE / f"{DATASET}_images"
    GT_JSON = HERE / f"{DATASET}_ground_truth.json"

    if args.reformat:
        gt = json.loads(GT_JSON.read_text())
        cat_names = {c["id"]: c["name"] for c in gt["categories"]}
        for name in args.models:
            flat = json.loads(pred_path(name).read_text())
            if flat and "prediction" in flat[0]:
                print(f"[{name}] already per-image format, skipping")
                continue
            pred_path(name).write_text(json.dumps(to_per_image(flat, gt, cat_names)))
            print(f"[{name}] reformatted {len(flat)} detections -> {pred_path(name)}")
        return

    missing = [str(weights_path(n)) for n in args.models if not weights_path(n).exists()]
    if missing:
        raise SystemExit("Trained weights not found (run train_bccd_detectors.py first):\n  " + "\n  ".join(missing))

    device = require_gpu()
    gt = json.loads(GT_JSON.read_text())
    cats = sorted(gt["categories"], key=lambda c: c["id"])
    cat_names = {c["id"]: c["name"] for c in cats}
    coco_gt = coco_from_dict(gt)
    PRED_DIR.mkdir(exist_ok=True)
    print(f"{len(gt['images'])} images in {IMAGE_DIR.name}")

    for name in args.models:
        print(f"\n[{name}] predicting with {weights_path(name)}")
        if MODEL_CONFIGS[name]["kind"] == "ultralytics":
            preds = predict_yolo(name, gt, cats, args.conf)
        else:
            preds = predict_tv(name, gt, cats, args.conf, device, args.workers)

        out = pred_path(name)
        out.write_text(json.dumps(to_per_image(preds, gt, cat_names)))
        print(f"[{name}] {len(preds)} detections -> {out}")
        # Note: includes the training images and the perturbed duplicates, so this is a
        # sanity check against bccdp_ground_truth.json, not a held-out score.
        print(f"[{name}] vs {DATASET} GT:{fmt_metrics(coco_metrics(coco_gt, preds, cat_names))}")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
