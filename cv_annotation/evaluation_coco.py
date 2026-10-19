#!/usr/bin/env python3

import os
import json
import argparse
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def load_labels(labels_path):
    with open(labels_path, "r") as f:
        labels = json.load(f)

    # Support common formats:
    #
    # {"0": "person", "1": "car", ...}
    #
    # or
    #
    # [{"id": 0, "name": "person"}, ...]
    #
    # or
    #
    # {"categories": [{"id": 1, "name": "person"}, ...]}

    if isinstance(labels, dict):
        if "categories" in labels:
            return {
                int(x["id"]): x["name"]
                for x in labels["categories"]
            }

        return {
            int(k): v
            for k, v in labels.items()
        }

    if isinstance(labels, list):
        return {
            int(x["id"]): x["name"]
            for x in labels
        }

    raise ValueError("Unsupported labels.json format")


def get_image_files(image_dir):
    extensions = {".jpg", ".jpeg", ".png", ".webp"}

    return sorted(
        p for p in Path(image_dir).iterdir()
        if p.suffix.lower() in extensions
    )


# ---------------------------------------------------------
# Faster R-CNN
# ---------------------------------------------------------

def load_fasterrcnn(model_path, num_classes, device, score_thresh, detections_per_img):
    # torchvision drops boxes below box_score_thresh (default 0.05) and keeps
    # at most detections_per_img (default 100) before we ever see them, so
    # both have to be set here to get low-confidence predictions.
    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=num_classes,
        box_score_thresh=score_thresh,
        box_detections_per_img=detections_per_img
    )

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model" in checkpoint:
            checkpoint = checkpoint["model"]

    checkpoint = {
        k.replace("module.", "", 1): v
        for k, v in checkpoint.items()
    }

    model.load_state_dict(
        checkpoint,
        strict=False
    )

    model.to(device)
    model.eval()

    return model


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main(args):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # -----------------------------------------------------
    # Labels
    # -----------------------------------------------------

    labels = load_labels(args.labels)

    # Make class list ordered by class ID.
    class_ids = sorted(labels.keys())
    class_names = [
        labels[class_id]
        for class_id in class_ids
    ]

    num_classes = len(class_names)

    print("Number of classes:", num_classes)

    # The detector outputs category IDs directly (0 = background), so its
    # head needs max_id + 1 outputs: 91 for COCO, matching torchvision's
    # COCO checkpoint.
    detector_num_classes = max(class_ids) + 1

    # -----------------------------------------------------
    # Images
    # -----------------------------------------------------

    image_files = get_image_files(args.images)

    print("Number of images:", len(image_files))

    # -----------------------------------------------------
    # Image preprocessing
    # -----------------------------------------------------

    detection_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # -----------------------------------------------------
    # Load Faster R-CNN
    # -----------------------------------------------------

    print("Loading Faster R-CNN...")

    fasterrcnn = load_fasterrcnn(
        args.fasterrcnn,
        detector_num_classes,
        device,
        args.detection_threshold,
        args.max_detections
    )

    # -----------------------------------------------------
    # Predictions
    # -----------------------------------------------------

    predictions = []

    for image_path in tqdm(image_files):

        try:
            image = Image.open(image_path).convert("RGB")

            # =================================================
            # Faster R-CNN
            # =================================================

            detection_input = detection_transform(image)
            detection_input = detection_input.to(device)

            with torch.no_grad():
                detection_output = fasterrcnn(
                    [detection_input]
                )[0]

            boxes = detection_output["boxes"]
            detection_labels = detection_output["labels"]
            detection_scores = detection_output["scores"]

            detections = []

            for box, label, score in zip(
                boxes,
                detection_labels,
                detection_scores
            ):

                score = float(score.item())

                if score < args.detection_threshold:
                    continue

                label = int(label.item())

                # Skip unknown label IDs.
                if label not in labels:
                    continue

                detections.append({
                    "class_id": label,
                    "class_name": labels[label],
                    "score": score,
                    "bbox": [
                        float(x)
                        for x in box.tolist()
                    ]
                })

            # =================================================
            # Store result
            # =================================================

            # Same layout as predictions.json, so find_label_issues.py
            # and the other scripts here can read it.
            predictions.append({
                "image_id": image_path.stem,
                "file_name": image_path.name,
                "prediction": detections
            })

        except Exception as e:

            print(
                f"\nError processing {image_path}: {e}"
            )

            predictions.append({
                "image_id": image_path.stem,
                "file_name": image_path.name,
                "prediction": [],
                "error": str(e)
            })

    # -----------------------------------------------------
    # Save JSON
    # -----------------------------------------------------

    # No indent: with low-confidence boxes kept this file is several
    # times larger than predictions.json.
    with open(args.output, "w") as f:
        json.dump(
            predictions,
            f
        )

    print()
    print("Saved predictions to:")
    print(args.output)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--images",
        default="data",
        help="COCO image directory, e.g. val2017/"
    )

    parser.add_argument(
        "--labels",
        default="labels.json",
        help="Path to labels.json"
    )

    parser.add_argument(
        "--fasterrcnn",
        default=os.path.join(
            "..", "..", "__models__",
            "fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
        ),
        help="Path to Faster R-CNN checkpoint"
    )

    parser.add_argument(
        "--output",
        default="predictions_low.json",
        help="Output JSON path"
    )

    # Low on purpose: cleanlab needs the low-confidence boxes too.
    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=0.01
    )

    parser.add_argument(
        "--max-detections",
        type=int,
        default=100,
        help="Max boxes kept per image"
    )

    args = parser.parse_args()

    main(args)