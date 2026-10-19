"""
Fine-tune three small pretrained object detectors on BCCD (RBC / WBC / Platelets).

Models (all COCO-pretrained, all < 100 MB):
    yolo11n   - Ultralytics YOLO11-nano                     (~5.4 MB)
    ssdlite   - torchvision SSDLite320 MobileNetV3-Large    (~13 MB)
    frcnn_mbv3- torchvision Faster R-CNN MobileNetV3-L FPN  (~74 MB)

Data:
    bccd_images/            *.jpg
    bccd_ground_truth.json  COCO-format ground truth (converted from bccd_annotations/*.xml)

The images are split deterministically into train/val (default 80/20). After every
epoch each model is evaluated on the val split and prints mAP50-95, mAP50, AR@100 and
per-class AP50. At the end every best checkpoint is re-scored with pycocotools on the
same val split so the three models are directly comparable.

Usage:
    python train_bccd_detectors.py                      # all three models
    python train_bccd_detectors.py --models yolo11n     # just one
    python train_bccd_detectors.py --epochs 10          # override epochs for all
"""

import argparse
import contextlib
import io
import json
import math
import os
import random
import shutil
import time
from collections import defaultdict
from functools import partial
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn as nn
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image


# ============================================================
# CONFIGURATION
# ============================================================

HERE = Path(__file__).resolve().parent
IMAGE_DIR = HERE / "bccd_images"
GT_JSON = HERE / "bccd_ground_truth.json"
OUTPUT_DIR = HERE / "bccd_training_runs"

MAX_MODEL_MB = 100

MODEL_CONFIGS = {
    "yolo11n": dict(kind="ultralytics", weights="yolo11n.pt", epochs=100, batch=16, imgsz=640),
    "ssdlite": dict(kind="torchvision", epochs=120, batch=16, lr=0.02, weight_decay=4e-5, input_size=512),
    "frcnn_mbv3": dict(kind="torchvision", epochs=40, batch=8, lr=0.01, weight_decay=1e-4),
}


# ============================================================
# UTILITIES
# ============================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def require_gpu():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not available - this script is meant to train on the GPU.")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    return torch.device("cuda:0")


def quiet(fn, *a, **kw):
    """Run a pycocotools call without its console spam."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def coco_from_dict(d):
    coco = COCO()
    coco.dataset = d
    quiet(coco.createIndex)
    return coco


def split_coco(gt, val_frac, seed):
    """Deterministic image-level train/val split of a COCO dict."""
    ids = sorted(img["id"] for img in gt["images"])
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_frac)))
    val_ids = set(ids[:n_val])

    def subset(keep):
        return {
            "info": gt.get("info", {}),
            "licenses": gt.get("licenses", []),
            "categories": gt["categories"],
            "images": [im for im in gt["images"] if im["id"] in keep],
            "annotations": [a for a in gt["annotations"] if a["image_id"] in keep],
        }

    train_ids = set(ids) - val_ids
    return subset(train_ids), subset(val_ids)


def coco_metrics(coco_gt, results, cat_names):
    """COCO bbox metrics for a list of detection results on coco_gt's images."""
    out = {"mAP50-95": 0.0, "mAP50": 0.0, "mAP75": 0.0, "AR100": 0.0,
           "AP50_per_class": {n: 0.0 for n in cat_names.values()}}
    if not results:
        return out
    coco_dt = quiet(coco_gt.loadRes, results)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.params.imgIds = sorted(coco_gt.getImgIds())
    quiet(ev.evaluate)
    quiet(ev.accumulate)
    quiet(ev.summarize)
    out.update({"mAP50-95": ev.stats[0], "mAP50": ev.stats[1], "mAP75": ev.stats[2], "AR100": ev.stats[8]})
    # precision: [IoU thr, recall, category, area range, max dets]
    prec = ev.eval["precision"]
    for k, cat_id in enumerate(ev.params.catIds):
        p = prec[0, :, k, 0, -1]
        p = p[p > -1]
        out["AP50_per_class"][cat_names[cat_id]] = float(p.mean()) if p.size else 0.0
    return out


def fmt_metrics(m):
    per_cls = " ".join(f"{n} {v:.3f}" for n, v in m["AP50_per_class"].items())
    return (f"val mAP50-95 {m['mAP50-95']:.3f}  mAP50 {m['mAP50']:.3f}  AR100 {m['AR100']:.3f}"
            f"  |  AP50 {per_cls}")


def file_mb(path):
    return os.path.getsize(path) / 1e6


# ============================================================
# PRETRAINED DOWNLOADS
# ============================================================

def tv_weights(name):
    from torchvision.models.detection import (
        FasterRCNN_MobileNet_V3_Large_FPN_Weights,
        SSDLite320_MobileNet_V3_Large_Weights,
    )
    return {
        "ssdlite": SSDLite320_MobileNet_V3_Large_Weights.COCO_V1,
        "frcnn_mbv3": FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1,
    }[name]


def download_pretrained(names, pretrained_dir):
    """Fetch every requested checkpoint up front and verify it is under MAX_MODEL_MB."""
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in names:
        cfg = MODEL_CONFIGS[name]
        if cfg["kind"] == "ultralytics":
            from ultralytics.utils.downloads import attempt_download_asset
            path = Path(attempt_download_asset(str(pretrained_dir / cfg["weights"])))
        else:
            w = tv_weights(name)
            w.get_state_dict(progress=True)  # downloads into the torch hub cache
            path = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(w.url).path).name
        size = file_mb(path)
        print(f"  {name:<11} {size:6.1f} MB  {path}")
        if size > MAX_MODEL_MB:
            raise SystemExit(f"{name} is {size:.1f} MB (> {MAX_MODEL_MB} MB limit)")
        paths[name] = path
    return paths


# ============================================================
# TORCHVISION MODELS
# ============================================================

class CocoDetection(Dataset):
    def __init__(self, coco_dict, image_dir, train):
        self.images = coco_dict["images"]
        self.image_dir = Path(image_dir)
        self.train = train
        self.anns = defaultdict(list)
        for a in coco_dict["annotations"]:
            self.anns[a["image_id"]].append(a)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        info = self.images[i]
        img = read_image(str(self.image_dir / info["file_name"]), mode=ImageReadMode.RGB).float() / 255.0
        _, h, w = img.shape

        anns = self.anns[info["id"]]
        boxes = torch.tensor([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2],
                               a["bbox"][1] + a["bbox"][3]] for a in anns], dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor([a["category_id"] for a in anns], dtype=torch.int64)

        if self.train:
            if random.random() < 0.5:
                img = img.flip(-1)
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            if random.random() < 0.5:
                img = img.flip(-2)
                boxes[:, [1, 3]] = h - boxes[:, [3, 1]]

        return img, {"boxes": boxes, "labels": labels, "image_id": info["id"]}


def collate(batch):
    return tuple(zip(*batch))


def build_torchvision_model(name, num_classes):
    """num_classes includes background (index 0); COCO category ids 1..N map directly to labels."""
    from torchvision.models import detection as det

    if name == "frcnn_mbv3":
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        model = det.fasterrcnn_mobilenet_v3_large_fpn(
            weights=tv_weights(name), box_score_thresh=0.001, box_detections_per_img=100)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    elif name == "ssdlite":
        from torchvision.models.detection import _utils as det_utils
        from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
        from torchvision.models.detection.transform import GeneralizedRCNNTransform
        model = det.ssdlite320_mobilenet_v3_large(weights=tv_weights(name), detections_per_img=100)
        in_channels = det_utils.retrieve_out_channels(model.backbone, (320, 320))
        num_anchors = model.anchor_generator.num_anchors_per_location()
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
        model.head.classification_head = SSDLiteClassificationHead(in_channels, num_anchors, num_classes, norm_layer)
        # Platelets are tiny at 320x320; anchors are relative so a larger input just works.
        s = MODEL_CONFIGS[name]["input_size"]
        model.transform = GeneralizedRCNNTransform(
            min_size=s, max_size=s, image_mean=model.transform.image_mean,
            image_std=model.transform.image_std, size_divisible=1, fixed_size=(s, s))
    else:
        raise ValueError(name)
    return model


@torch.no_grad()
def predict_torchvision(model, loader, device):
    model.eval()
    results = []
    for imgs, targets in loader:
        imgs = [im.to(device, non_blocking=True) for im in imgs]
        with torch.autocast("cuda", dtype=torch.float16):
            outs = model(imgs)
        for t, o in zip(targets, outs):
            boxes = o["boxes"].float().cpu()
            boxes[:, 2:] -= boxes[:, :2]
            for b, s, l in zip(boxes.tolist(), o["scores"].float().cpu().tolist(), o["labels"].cpu().tolist()):
                results.append({"image_id": t["image_id"], "category_id": int(l),
                                "bbox": [round(v, 2) for v in b], "score": round(s, 5)})
    return results


def train_torchvision(name, train_dict, val_dict, coco_val, cat_names, args, device, out_dir):
    cfg = MODEL_CONFIGS[name]
    epochs = args.epochs or cfg["epochs"]
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_torchvision_model(name, num_classes=len(cat_names) + 1).to(device)

    train_loader = DataLoader(CocoDetection(train_dict, IMAGE_DIR, train=True), batch_size=cfg["batch"],
                              shuffle=True, num_workers=args.workers, collate_fn=collate,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(CocoDetection(val_dict, IMAGE_DIR, train=False), batch_size=cfg["batch"],
                            shuffle=False, num_workers=args.workers, collate_fn=collate, pin_memory=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=cfg["lr"], momentum=0.9, weight_decay=cfg["weight_decay"])
    total_iters = epochs * len(train_loader)
    warmup = min(300, 3 * len(train_loader))

    def lr_lambda(it):
        if it < warmup:
            return 0.001 + 0.999 * it / warmup
        p = (it - warmup) / max(1, total_iters - warmup)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda")

    best, history = -1.0, []
    print(f"\n[{name}] training {epochs} epochs on {len(train_loader.dataset)} images "
          f"(val {len(val_loader.dataset)})")
    for epoch in range(1, epochs + 1):
        model.train()
        t0, loss_sum, n = time.time(), 0.0, 0
        for imgs, targets in train_loader:
            imgs = [im.to(device, non_blocking=True) for im in imgs]
            targets = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]
            with torch.autocast("cuda", dtype=torch.float16):
                loss = sum(model(imgs, targets).values())
            if not torch.isfinite(loss):
                print(f"[{name}] non-finite loss, skipping batch")
                opt.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            loss_sum += loss.item()
            n += 1

        m = coco_metrics(coco_val, predict_torchvision(model, val_loader, device), cat_names)
        history.append({"epoch": epoch, "train_loss": loss_sum / max(1, n), **m})
        torch.save(model.state_dict(), out_dir / "last.pt")
        if m["mAP50-95"] > best:
            best = m["mAP50-95"]
            torch.save(model.state_dict(), out_dir / "best.pt")
        print(f"[{name}] epoch {epoch:3d}/{epochs}  loss {loss_sum / max(1, n):.4f}  |  {fmt_metrics(m)}"
              f"  (best mAP50-95 {best:.3f})  {time.time() - t0:.1f}s")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    return predict_torchvision(model, val_loader, device)


# ============================================================
# ULTRALYTICS YOLO
# ============================================================

def build_yolo_dataset(train_dict, val_dict, cats, root):
    """Write an Ultralytics-style dataset (images symlinked, labels as normalised xywh txt)."""
    if root.exists():
        shutil.rmtree(root)
    cat_to_idx = {c["id"]: i for i, c in enumerate(cats)}
    for split, d in (("train", train_dict), ("val", val_dict)):
        img_dir, lbl_dir = root / "images" / split, root / "labels" / split
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        anns = defaultdict(list)
        for a in d["annotations"]:
            anns[a["image_id"]].append(a)
        for im in d["images"]:
            src, dst = (IMAGE_DIR / im["file_name"]).resolve(), img_dir / im["file_name"]
            try:
                os.symlink(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            W, H = im["width"], im["height"]
            lines = []
            for a in anns[im["id"]]:
                x, y, w, h = a["bbox"]
                lines.append(f"{cat_to_idx[a['category_id']]} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} "
                             f"{w / W:.6f} {h / H:.6f}")
            (lbl_dir / (Path(im["file_name"]).stem + ".txt")).write_text("\n".join(lines))

    yaml_path = root / "data.yaml"
    names = "\n".join(f"  {i}: {c['name']}" for i, c in enumerate(cats))
    yaml_path.write_text(f"path: {root.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    return yaml_path


def train_yolo(name, weights_path, train_dict, val_dict, cats, args, out_dir):
    from ultralytics import YOLO

    cfg = MODEL_CONFIGS[name]
    epochs = args.epochs or cfg["epochs"]
    data_yaml = build_yolo_dataset(train_dict, val_dict, cats, OUTPUT_DIR / "yolo_dataset")

    model = YOLO(str(weights_path))

    def on_epoch_end(trainer):
        m = trainer.metrics
        print(f"[{name}] epoch {trainer.epoch + 1:3d}/{trainer.epochs}  |  "
              f"val mAP50-95 {m.get('metrics/mAP50-95(B)', 0):.3f}  mAP50 {m.get('metrics/mAP50(B)', 0):.3f}  "
              f"P {m.get('metrics/precision(B)', 0):.3f}  R {m.get('metrics/recall(B)', 0):.3f}  "
              f"(best fitness {trainer.best_fitness or 0:.3f})")

    model.add_callback("on_fit_epoch_end", on_epoch_end)

    print(f"\n[{name}] training {epochs} epochs on {len(train_dict['images'])} images "
          f"(val {len(val_dict['images'])})")
    model.train(data=str(data_yaml), epochs=epochs, imgsz=cfg["imgsz"], batch=cfg["batch"], device=0,
                workers=args.workers, seed=args.seed, project=str(out_dir.parent), name=out_dir.name,
                exist_ok=True, verbose=True)

    best = YOLO(str(out_dir / "weights" / "best.pt"))
    idx_to_cat = {i: c["id"] for i, c in enumerate(cats)}
    results = []
    val_dir = OUTPUT_DIR / "yolo_dataset" / "images" / "val"
    for im in val_dict["images"]:
        r = best.predict(str(val_dir / im["file_name"]), imgsz=cfg["imgsz"], conf=0.001, iou=0.7,
                         max_det=100, device=0, verbose=False)[0]
        xyxy = r.boxes.xyxy.cpu().numpy()
        for (x1, y1, x2, y2), s, c in zip(xyxy, r.boxes.conf.cpu().tolist(), r.boxes.cls.cpu().tolist()):
            results.append({"image_id": im["id"], "category_id": idx_to_cat[int(c)],
                            "bbox": [round(float(x1), 2), round(float(y1), 2),
                                     round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                            "score": round(s, 5)})
    return results


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODEL_CONFIGS), choices=list(MODEL_CONFIGS))
    ap.add_argument("--epochs", type=int, default=None, help="override per-model default epochs")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = require_gpu()

    gt = json.loads(GT_JSON.read_text())
    cats = sorted(gt["categories"], key=lambda c: c["id"])
    cat_names = {c["id"]: c["name"] for c in cats}
    train_dict, val_dict = split_coco(gt, args.val_frac, args.seed)

    splits_dir = OUTPUT_DIR / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "train.json").write_text(json.dumps(train_dict))
    (splits_dir / "val.json").write_text(json.dumps(val_dict))
    coco_val = coco_from_dict(val_dict)
    print(f"Split: {len(train_dict['images'])} train / {len(val_dict['images'])} val images "
          f"-> {splits_dir}")

    print("\nDownloading pretrained weights:")
    weights = download_pretrained(args.models, OUTPUT_DIR / "pretrained")

    summary = {}
    for name in args.models:
        seed_everything(args.seed)
        out_dir = OUTPUT_DIR / name
        if MODEL_CONFIGS[name]["kind"] == "ultralytics":
            preds = train_yolo(name, weights[name], train_dict, val_dict, cats, args, out_dir)
        else:
            preds = train_torchvision(name, train_dict, val_dict, coco_val, cat_names, args, device, out_dir)

        (out_dir / "predictions_val.json").write_text(json.dumps(preds))
        m = coco_metrics(coco_val, preds, cat_names)
        summary[name] = m
        print(f"\n[{name}] BEST CHECKPOINT (pycocotools, val split)  {fmt_metrics(m)}")
        torch.cuda.empty_cache()

    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 78)
    print(f"{'model':<12}{'mAP50-95':>10}{'mAP50':>9}{'mAP75':>9}{'AR100':>9}   "
          + " ".join(f"{n:>9}" for n in cat_names.values()))
    for name, m in summary.items():
        print(f"{name:<12}{m['mAP50-95']:>10.3f}{m['mAP50']:>9.3f}{m['mAP75']:>9.3f}{m['AR100']:>9.3f}   "
              + " ".join(f"{v:>9.3f}" for v in m["AP50_per_class"].values()))
    print(f"(per-class columns are AP50)  Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
