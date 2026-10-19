#!/usr/bin/env python3
"""
Run every usable model in the FiftyOne __models__ folder over the images and write
one predictions file per model, in the same layout as predictions.json:

    [{"image_id": stem, "file_name": ..., "prediction": [
        {"class_id": COCO id, "class_name": ..., "score": ..., "bbox": [x1, y1, x2, y2]}]}]

  rcnn    fasterrcnn_resnet50_fpn_coco-*.pth -> predictions_rcnn.json
          COCO detector; low threshold so cleanlab also sees the low-confidence boxes
          (same settings as evaluation_coco.py / predictions_low.json).
  yolo    yolo11n.pt -> predictions_yolo.json
          Ultralytics YOLO11n: a tiny COCO detector (2.6M params, 39.5 mAP), same
          low threshold. Needs `pip install ultralytics`.
  yolo11s/m/l/x  yolo11<size>.pt -> predictions_yolo11<size>.json: the larger YOLO11 models,
          same code (--models yolo11x).
  clip    CLIP-ViT-B-32.pt -> predictions_clip.json
          CLIP has no detector, so every annotated box in labels.json is cropped and
          zero-shot classified into the COCO classes. The boxes are the annotated boxes,
          so in cleanlab these predictions can only reveal wrong classes (swap), never
          missing boxes (overlooked) or badly placed ones (badloc).
  resnet  resnet18-*.pth -> skipped: an ImageNet classifier, it has no COCO classes
          and no boxes.

A model whose weights file is missing is skipped.
"""

import os
import json
import glob
import argparse
from concurrent.futures import ThreadPoolExecutor

import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from evaluation_coco import load_labels, get_image_files, load_fasterrcnn


def find_weights(models_dir, pattern):
    matches = sorted(glob.glob(os.path.join(models_dir, pattern)))
    return matches[0] if matches else None


def write_predictions(path, predictions):
    # No indent: with low-confidence boxes kept these files get large.
    with open(path, "w") as f:
        json.dump(predictions, f)
    n = sum(len(p["prediction"]) for p in predictions)
    print(f"Wrote {path}  ({len(predictions)} images, {n} predictions)")


# ---------------------------------------------------------
# Faster R-CNN: detect boxes in every image
# ---------------------------------------------------------

def predict_rcnn(weights, image_files, labels, device, args):
    model = load_fasterrcnn(weights, max(labels) + 1, device,
                            args.detection_threshold, args.max_detections)
    to_tensor = transforms.ToTensor()
    predictions = []
    for image_path in tqdm(image_files, desc="rcnn"):
        entry = {"image_id": image_path.stem, "file_name": image_path.name, "prediction": []}
        try:
            image = Image.open(image_path).convert("RGB")
            with torch.no_grad():
                out = model([to_tensor(image).to(device)])[0]
            for box, label, score in zip(out["boxes"], out["labels"], out["scores"]):
                label, score = int(label), float(score)
                if score < args.detection_threshold or label not in labels:
                    continue
                entry["prediction"].append({"class_id": label, "class_name": labels[label],
                                            "score": score, "bbox": [float(x) for x in box.tolist()]})
        except Exception as e:
            print(f"\nError processing {image_path}: {e}")
            entry["error"] = str(e)
        predictions.append(entry)
    return predictions


# ---------------------------------------------------------
# YOLO11n (ultralytics): small COCO detector
# ---------------------------------------------------------

def predict_yolo(weights, image_files, labels, device, args):
    from ultralytics import YOLO

    model = YOLO(weights)
    # YOLO numbers the 80 COCO classes 0..79; map them back to COCO ids by name
    name_to_id = {n: c for c, n in labels.items()}
    yolo_to_coco = {i: name_to_id.get(n) for i, n in model.names.items()}
    missing = [model.names[i] for i, c in yolo_to_coco.items() if c is None]
    if missing:
        print(f"[yolo] classes not in labels.json, dropped: {missing}")

    predictions = []
    for image_path in tqdm(image_files, desc="yolo"):
        entry = {"image_id": image_path.stem, "file_name": image_path.name, "prediction": []}
        try:
            boxes = model.predict(str(image_path), conf=args.detection_threshold, max_det=args.max_detections,
                                  device=str(device), verbose=False)[0].boxes
            for box, cls, score in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist()):
                c = yolo_to_coco.get(int(cls))
                if c is None:
                    continue
                entry["prediction"].append({"class_id": c, "class_name": labels[c],
                                            "score": float(score), "bbox": [float(x) for x in box]})
        except Exception as e:
            print(f"\nError processing {image_path}: {e}")
            entry["error"] = str(e)
        predictions.append(entry)
    return predictions


# ---------------------------------------------------------
# CLIP: zero-shot classify every annotated box
# ---------------------------------------------------------

def predict_clip(weights, image_files, labels, coco, device, args):
    import open_clip

    # fp32: the JIT checkpoint loads as fp16 on GPU, which breaks the fp32 preprocessing
    model = open_clip.load_openai_model(weights, precision="fp32", device=device)
    model.eval()
    preprocess = open_clip.image_transform(model.visual.image_size, is_train=False)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    class_ids = sorted(labels)
    with torch.no_grad():
        text = tokenizer([f"a photo of a {labels[c]}." for c in class_ids]).to(device)
        text_feats = model.encode_text(text)
        text_feats /= text_feats.norm(dim=-1, keepdim=True)
        logit_scale = model.logit_scale.exp()

    images_by_name = {im["file_name"]: im for im in coco["images"]}
    anns_by_image = {}
    for a in coco["annotations"]:
        # crowd boxes cover groups of objects; cleanlab ignores them too
        if not a.get("iscrowd", 0) and a["bbox"][2] > 0 and a["bbox"][3] > 0:
            anns_by_image.setdefault(a["image_id"], []).append(a)

    def crops_for(image_path):
        """(boxes in xyxy, preprocessed crops) for the annotations of one image."""
        image = images_by_name.get(image_path.name)
        anns = anns_by_image.get(image["id"], []) if image else []
        if not anns:
            return [], []
        im = Image.open(image_path).convert("RGB")
        boxes, crops = [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            # a little context around the box helps CLIP recognise small objects
            px, py = w * args.clip_pad, h * args.clip_pad
            crop = im.crop((max(0, x - px), max(0, y - py),
                            min(im.width, x + w + px), min(im.height, y + h + py)))
            boxes.append([x, y, x + w, y + h])
            crops.append(preprocess(crop))
        return boxes, crops

    def safe_crops(image_path):
        try:
            return crops_for(image_path), None
        except Exception as e:
            return ([], []), str(e)

    predictions = []
    pending = []  # (prediction entry, box) per crop waiting in the batch
    batch = []

    def flush():
        if not batch:
            return
        with torch.no_grad():
            feats = model.encode_image(torch.stack(batch).to(device))
            feats /= feats.norm(dim=-1, keepdim=True)
            probs = (logit_scale * feats @ text_feats.T).softmax(dim=-1)
        top_p, top_i = probs.topk(args.clip_topk, dim=-1)
        for (entry, box), ps, idx in zip(pending, top_p.tolist(), top_i.tolist()):
            for p, i in zip(ps, idx):
                if p >= args.clip_min_score:
                    c = class_ids[i]
                    entry["prediction"].append({"class_id": c, "class_name": labels[c],
                                                "score": float(p), "bbox": [float(v) for v in box]})
        pending.clear()
        batch.clear()

    with ThreadPoolExecutor(args.workers) as pool:
        results = pool.map(safe_crops, image_files)
        for image_path, ((boxes, crops), error) in tqdm(zip(image_files, results),
                                                        total=len(image_files), desc="clip"):
            entry = {"image_id": image_path.stem, "file_name": image_path.name, "prediction": []}
            if error:
                print(f"\nError processing {image_path}: {error}")
                entry["error"] = error
            predictions.append(entry)
            for box, crop in zip(boxes, crops):
                pending.append((entry, box))
                batch.append(crop)
                if len(batch) >= args.clip_batch:
                    flush()
        flush()
    return predictions


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

MODELS = {
    # name: (weights glob, output file)
    "rcnn":   ("fasterrcnn_resnet50_fpn_coco*.pth", "predictions_rcnn.json"),
    "yolo":   ("yolo11n*.pt",                       "predictions_yolo.json"),
    # larger YOLO11 sizes (COCO mAP: s 47.0, m 51.5, l 53.4, x 54.7), same code path as yolo
    "yolo11s": ("yolo11s*.pt",                      "predictions_yolo11s.json"),
    "yolo11m": ("yolo11m*.pt",                      "predictions_yolo11m.json"),
    "yolo11l": ("yolo11l*.pt",                      "predictions_yolo11l.json"),
    "yolo11x": ("yolo11x*.pt",                      "predictions_yolo11x.json"),
    "clip":   ("CLIP-ViT-B-32*.pt",                 "predictions_clip.json"),
    "resnet": ("resnet18*.pth",                     "predictions_resnet.json"),
}


def main(args):
    device = torch.device(args.device)
    print("Device:", device)

    with open(args.labels) as f:
        coco = json.load(f)
    labels = load_labels(args.labels)
    image_files = get_image_files(args.images)
    print(f"{len(image_files)} images, {len(labels)} classes")

    for name in args.models.split(","):
        name = name.strip()
        if name not in MODELS:
            print(f"\n[{name}] unknown model, skipped")
            continue
        pattern, out_name = MODELS[name]
        weights = find_weights(args.models_dir, pattern)
        if not weights:
            print(f"\n[{name}] no weights matching {pattern} in {args.models_dir}, skipped")
            continue
        out_path = os.path.join(args.out_dir, out_name)
        print(f"\n[{name}] {weights}")

        if name == "rcnn":
            write_predictions(out_path, predict_rcnn(weights, image_files, labels, device, args))
        elif name.startswith("yolo"):
            write_predictions(out_path, predict_yolo(weights, image_files, labels, device, args))
        elif name == "clip":
            write_predictions(out_path, predict_clip(weights, image_files, labels, coco, device, args))
        elif name == "resnet":
            print(f"[{name}] skipped: ImageNet classifier (1000 ImageNet classes, no boxes), "
                  "so it can't make COCO detections")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", default="data", help="Image directory")
    parser.add_argument("--labels", default="labels.json", help="COCO labels (classes; boxes for CLIP)")
    parser.add_argument("--models-dir", default=os.path.join(os.path.expanduser("~"), "fiftyone", "__models__"))
    parser.add_argument("--models", default="rcnn,yolo,clip,resnet", help="Comma-separated models to run")
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Faster R-CNN. Low on purpose: cleanlab needs the low-confidence boxes too.
    parser.add_argument("--detection-threshold", type=float, default=0.01)
    parser.add_argument("--max-detections", type=int, default=100, help="Max boxes kept per image")
    # CLIP
    parser.add_argument("--clip-topk", type=int, default=3, help="Classes kept per box")
    parser.add_argument("--clip-min-score", type=float, default=0.01, help="Drop classes below this probability")
    parser.add_argument("--clip-pad", type=float, default=0.1, help="Context added around each box, as a fraction of its size")
    parser.add_argument("--clip-batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8, help="Threads loading and cropping images")
    main(parser.parse_args())
