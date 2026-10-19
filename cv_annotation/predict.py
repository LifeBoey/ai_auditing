"""
predict.py -- run one or more trained object detectors over a folder of images and write one
predictions JSON per model.

Models (--model ARCH WEIGHTS [key=value ...], repeat for several):
    ARCH = ultralytics        any Ultralytics checkpoint (YOLOv5u/8/11/12/26, RT-DETR, ...)
    ARCH = <torchvision name> fasterrcnn_resnet50_fpn[_v2], fasterrcnn_mobilenet_v3_large[_320]_fpn,
                              retinanet_resnet50_fpn[_v2], fcos_resnet50_fpn, ssd300_vgg16,
                              ssdlite320_mobilenet_v3_large, maskrcnn_resnet50_fpn[_v2]
                              (aliases: frcnn_mbv3, ssdlite)
    WEIGHTS                   a checkpoint file: a state_dict, a {"model"/"state_dict": ...} dict, or
                              a whole pickled model. For torchvision, `coco` loads the stock
                              COCO-pretrained weights instead.
    key=value options         name=<output name>     (default: the checkpoint's run folder, or ARCH)
                              size=<px>              input size (Ultralytics imgsz; SSD/SSDLite fixed
                                                     input size, e.g. size=512 for the BCCD ssdlite)
                              num_classes=<n>        torchvision, incl. background (default: max
                                                     category id in --labels + 1)
                              label_map=id|index     torchvision label -> category: the label IS the
                                                     category id (default), or the i-th category

Images and ids:
    --labels <COCO json>  images, image ids and category names come from it (recommended: the
                          output then lines up with that ground truth). Ultralytics classes are
                          matched to its categories by name.
    no --labels           every image file in --images, ids 1..N in file-name order; classes keep
                          the model's own ids and names.

Output (--out-dir, one file per model):  predictions_<prefix>_<name>.json
    --format per_image (default; what flag_annotations.py, blackbox_*.py, phoenix_slices.py and
                        coco_review_slideshow.py read):
        [{"image_id", "file_name", "prediction": [{"class_id", "class_name", "score",
                                                   "bbox": [x1, y1, x2, y2]}]}]
    --format coco      flat COCO results: [{"image_id", "category_id", "bbox": [x, y, w, h], "score"}]
Keep --conf low (default 0.001) for mAP and audits; filter later.

Example:
    python predict.py --images ../DATA/bccdp_images --labels ../DATA/bccdp_ground_truth.json \\
        --model ultralytics ../DATA/bccd_training_runs/yolo11n/weights/best.pt name=yolo11n \\
        --model frcnn_mbv3 ../DATA/bccd_training_runs/frcnn_mbv3/best.pt \\
        --model ssdlite ../DATA/bccd_training_runs/ssdlite/best.pt size=512 \\
        --out-dir ../DATA/prediction_jsons --prefix bccdp
"""

import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path

import torch

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ALIASES = {"frcnn_mbv3": "fasterrcnn_mobilenet_v3_large_fpn", "ssdlite": "ssdlite320_mobilenet_v3_large"}
# torchvision builders take their score threshold / max detections under different names
RCNN_KW = ("box_score_thresh", "box_detections_per_img")
DENSE_KW = ("score_thresh", "detections_per_img")
TV_ARCHS = {
    "fasterrcnn_resnet50_fpn": RCNN_KW, "fasterrcnn_resnet50_fpn_v2": RCNN_KW,
    "fasterrcnn_mobilenet_v3_large_fpn": RCNN_KW, "fasterrcnn_mobilenet_v3_large_320_fpn": RCNN_KW,
    "maskrcnn_resnet50_fpn": RCNN_KW, "maskrcnn_resnet50_fpn_v2": RCNN_KW,
    "retinanet_resnet50_fpn": DENSE_KW, "retinanet_resnet50_fpn_v2": DENSE_KW,
    "fcos_resnet50_fpn": DENSE_KW, "ssd300_vgg16": DENSE_KW, "ssdlite320_mobilenet_v3_large": DENSE_KW,
}


# ============================================================
# images + categories
# ============================================================
def load_images(images_dir, labels_path):
    """(images [{id, file_name}], categories {id: name} or None). Images missing on disk are dropped."""
    images_dir = Path(images_dir)
    if labels_path:
        coco = json.loads(Path(labels_path).read_text(encoding="utf-8"))
        images = [im for im in coco["images"] if (images_dir / im["file_name"]).is_file()]
        if len(images) < len(coco["images"]):
            print(f"[warn] {len(coco['images']) - len(images)} images in {labels_path} not found in {images_dir}, skipped")
        cats = {c["id"]: c["name"] for c in coco.get("categories", [])}
        return images, cats
    files = sorted(p.name for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return [{"id": i, "file_name": f} for i, f in enumerate(files, start=1)], None


def parse_model(spec):
    if len(spec) < 2:
        sys.exit(f"--model needs ARCH WEIGHTS [key=value ...], got: {' '.join(spec)}")
    arch, weights, *opts = spec
    arch = ALIASES.get(arch, arch)
    if arch != "ultralytics" and arch not in TV_ARCHS:
        sys.exit(f"Unknown ARCH '{spec[0]}'. Use 'ultralytics' or one of: {', '.join(sorted(TV_ARCHS))}")
    kv = {}
    for o in opts:
        if "=" not in o:
            sys.exit(f"Model option '{o}' is not key=value")
        k, v = o.split("=", 1)
        kv[k] = v
    if "name" not in kv:
        p = Path(weights)
        # .../<run>/best.pt or .../<run>/weights/best.pt -> <run>
        run = p.parent.parent.name if p.parent.name == "weights" else p.parent.name
        kv["name"] = run if weights != "coco" and run not in ("", ".") else spec[0]
    return dict(arch=arch, weights=weights, **kv)


# ============================================================
# Ultralytics
# ============================================================
def predict_ultralytics(m, images, images_dir, cats, args):
    from ultralytics import YOLO

    model = YOLO(m["weights"])
    names = model.names                       # {index: name}
    if cats:
        by_name = {n.lower(): cid for cid, n in cats.items()}
        if all(n.lower() in by_name for n in names.values()):
            idx_to_cat = {i: by_name[n.lower()] for i, n in names.items()}
        elif len(names) == len(cats):
            print(f"[warn] {m['name']}: class names {list(names.values())} don't match the labels' "
                  f"{list(cats.values())}; mapping by order of category id")
            idx_to_cat = dict(zip(sorted(names), sorted(cats)))
        else:
            sys.exit(f"{m['name']}: model has {len(names)} classes {list(names.values())}, labels have "
                     f"{len(cats)} {list(cats.values())}; can't map them")
        cat_name = cats
    else:
        idx_to_cat, cat_name = {i: i for i in names}, dict(names)

    kw = dict(conf=args.conf, iou=0.7, max_det=args.max_det, device=args.device, verbose=False)
    if "size" in m:
        kw["imgsz"] = int(m["size"])
    out = {}
    for s in range(0, len(images), args.batch):
        batch = images[s:s + args.batch]
        for im, r in zip(batch, model.predict([str(Path(images_dir) / im["file_name"]) for im in batch], **kw)):
            out[im["id"]] = [
                dict(class_id=idx_to_cat[int(c)], class_name=cat_name[idx_to_cat[int(c)]], score=round(float(sc), 5),
                     bbox=[round(float(v), 2) for v in b])
                for b, sc, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
        print(f"\r  {m['name']}: {min(s + args.batch, len(images))}/{len(images)} images", end="")
    print()
    return out, cat_name


# ============================================================
# torchvision
# ============================================================
class ImageFolder(torch.utils.data.Dataset):
    def __init__(self, images, images_dir):
        self.images, self.dir = images, Path(images_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        from torchvision.io import ImageReadMode, read_image
        im = self.images[i]
        return read_image(str(self.dir / im["file_name"]), mode=ImageReadMode.RGB).float() / 255.0, im["id"]


def collate(batch):
    return tuple(zip(*batch))


def build_torchvision(m, cats, args):
    from torchvision.models import detection

    arch = m["arch"]
    builder = getattr(detection, arch)
    thresh_kw, det_kw = TV_ARCHS[arch]
    kw = {thresh_kw: args.conf, det_kw: args.max_det}

    if m["weights"] == "coco":
        weights = detection.get_model_weights(arch).DEFAULT
        model = builder(weights=weights, **kw)
        names = dict(enumerate(weights.meta["categories"]))
    else:
        if "num_classes" in m:
            num_classes = int(m["num_classes"])
        elif cats:
            num_classes = (len(cats) if m.get("label_map") == "index" else max(cats)) + 1
        else:
            sys.exit(f"{m['name']}: pass --labels or num_classes=<n> (including background)")
        model = builder(weights=None, weights_backbone=None, num_classes=num_classes, **kw)
        ckpt = torch.load(m["weights"], map_location="cpu", weights_only=False)   # trusted checkpoints only
        if isinstance(ckpt, torch.nn.Module):
            model = ckpt
        else:
            state = next((ckpt[k] for k in ("model", "state_dict", "model_state_dict") if k in ckpt), ckpt)
            try:
                missing, unexpected = model.load_state_dict(state, strict=False)
            except RuntimeError as e:
                sys.exit(f"{m['name']}: checkpoint doesn't fit {arch} with num_classes={num_classes} "
                         f"(check ARCH / num_classes): {e}")
            # Fine-tuned checkpoints use FrozenBatchNorm (no num_batches_tracked); built from scratch the
            # model has plain BatchNorm. Identical at inference, so only those buffers may differ.
            bad = [k for k in missing + unexpected if not k.endswith("num_batches_tracked")]
            if bad:
                sys.exit(f"{m['name']}: checkpoint doesn't fit {arch} with num_classes={num_classes} "
                         f"(check ARCH / num_classes): {bad[:5]}{' ...' if len(bad) > 5 else ''}")
        names = None

    if "size" in m and arch.startswith("ssd"):
        from torchvision.models.detection.transform import GeneralizedRCNNTransform
        s = int(m["size"])
        t = model.transform
        model.transform = GeneralizedRCNNTransform(s, s, t.image_mean, t.image_std, size_divisible=1, fixed_size=(s, s))
    return model, names


@torch.no_grad()
def predict_torchvision(m, images, images_dir, cats, args):
    model, coco_names = build_torchvision(m, cats, args)
    model.to(args.device).eval()

    if coco_names is not None:                # stock COCO weights: labels are COCO category ids
        label_to_cat = lambda l: l
        cat_name = cats or {i: n for i, n in coco_names.items() if n != "N/A"}
    elif cats and m.get("label_map") == "index":
        order = sorted(cats)
        label_to_cat = lambda l: order[l - 1]
        cat_name = cats
    else:
        label_to_cat = lambda l: l
        cat_name = cats or {}

    loader = torch.utils.data.DataLoader(ImageFolder(images, images_dir), batch_size=args.batch,
                                         num_workers=args.workers, collate_fn=collate)
    out, done = {}, 0
    for imgs, ids in loader:
        with torch.autocast("cuda", enabled=args.amp and args.device.startswith("cuda")):
            res = model([im.to(args.device) for im in imgs])
        for iid, r in zip(ids, res):
            preds = []
            for b, sc, l in zip(r["boxes"].float().tolist(), r["scores"].float().tolist(), r["labels"].tolist()):
                if l == 0 or sc < args.conf:
                    continue
                cid = label_to_cat(int(l))
                if cats and cid not in cats:
                    continue                  # e.g. a stock COCO class the labels don't have
                preds.append(dict(class_id=cid, class_name=cat_name.get(cid, f"class_{cid}"),
                                  score=round(sc, 5), bbox=[round(v, 2) for v in b]))
            out[iid] = preds
        done += len(ids)
        print(f"\r  {m['name']}: {done}/{len(images)} images", end="")
    print()
    return out, cat_name


# ============================================================
# output + optional evaluation
# ============================================================
def to_output(images, preds, fmt):
    if fmt == "coco":
        return [dict(image_id=im["id"], category_id=p["class_id"], score=p["score"],
                     bbox=[p["bbox"][0], p["bbox"][1], round(p["bbox"][2] - p["bbox"][0], 2),
                           round(p["bbox"][3] - p["bbox"][1], 2)])
                for im in images for p in preds.get(im["id"], [])]
    return [dict(image_id=im["id"], file_name=im["file_name"],
                 prediction=sorted(preds.get(im["id"], []), key=lambda p: -p["score"])) for im in images]


def evaluate(labels_path, images, preds):
    """COCO bbox mAP over the predicted images (needs pycocotools); None if unavailable."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        return None
    results = to_output(images, preds, "coco")
    if not results:
        return 0.0, 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(labels_path)
        ev = COCOeval(gt, gt.loadRes(results), "bbox")
        ev.params.imgIds = [im["id"] for im in images]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return ev.stats[0], ev.stats[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="folder of images")
    ap.add_argument("--labels", default=None, help="COCO json supplying image ids + categories (recommended)")
    ap.add_argument("--model", nargs="+", action="append", required=True, metavar="ARCH WEIGHTS [key=value]",
                    help="repeat for several models; see the module docstring")
    ap.add_argument("--out-dir", default="predictions")
    ap.add_argument("--prefix", default=None, help="output name: predictions_<prefix>_<model>.json "
                                                   "(default: the images folder name minus '_images')")
    ap.add_argument("--format", choices=("per_image", "coco"), default="per_image")
    ap.add_argument("--conf", type=float, default=0.001, help="lowest score kept")
    ap.add_argument("--max-det", type=int, default=100, help="most boxes kept per image")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true", help="fp16 autocast for torchvision models on GPU")
    ap.add_argument("--no-eval", action="store_true", help="skip the mAP printout against --labels")
    args = ap.parse_args()

    models = [parse_model(spec) for spec in args.model]
    names = [m["name"] for m in models]
    if len(set(names)) < len(names):
        sys.exit(f"Two models would write the same file ({names}); give them name=... options")
    if args.device == "cpu":
        print("[warn] no CUDA GPU found, running on CPU")

    images, cats = load_images(args.images, args.labels)
    if not images:
        sys.exit(f"No images found in {args.images}")
    prefix = args.prefix or Path(args.images.rstrip("/\\")).name.removesuffix("_images")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(images)} images from {args.images}" + (f", ids/categories from {args.labels}" if args.labels else ""))

    for m in models:
        print(f"\n[{m['name']}] {m['arch']}  {m['weights']}")
        run = predict_ultralytics if m["arch"] == "ultralytics" else predict_torchvision
        preds, _ = run(m, images, args.images, cats, args)
        path = out_dir / f"predictions_{prefix}_{m['name']}.json"
        path.write_text(json.dumps(to_output(images, preds, args.format)))
        print(f"[{m['name']}] {sum(map(len, preds.values()))} detections -> {path}")
        if args.labels and not args.no_eval:
            r = evaluate(args.labels, images, preds)
            if r:
                print(f"[{m['name']}] vs {Path(args.labels).name}: mAP50-95 {r[0]:.3f}  mAP50 {r[1]:.3f}")
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
