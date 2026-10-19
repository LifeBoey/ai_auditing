
import json
import math
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_DIR = Path("data_sub500_v2_metadata")
GT_JSON = Path("labels_sub500_v2.json")
PREDICTIONS_JSON = Path("prediction_jsons/predictions_coco.json")

METADATA_CLASSES = [
    "brightness",
    # "weather",
    # "scene_type",
]

IOU_TYPE = "bbox"  # "bbox" or "segm"
OUTPUT_CSV = Path("metadata_evaluation.csv")

# How to group images with absent metadata fields:
MISSING_VALUE = "MISSING"

import json
from pathlib import Path
from PIL import Image, ExifTags

# IMPORTANT: use the directory containing the PROCESSED images.
MAX_IMAGES_TO_CHECK = 10

USER_COMMENT_TAG = 37510

with open(GT_JSON, "r") as f:
    gt = json.load(f)

images = gt["images"]

print("=== DATASET SUMMARY ===")
print("COCO images:", len(images))
print("Image directory:", IMAGE_DIR.resolve())
print("Directory exists:", IMAGE_DIR.is_dir())
print("Images in directory:",
      len(list(IMAGE_DIR.iterdir())) if IMAGE_DIR.is_dir() else 0)

print("\n=== COCO IMAGE ENTRY ===")
if images:
    print(json.dumps(images[0], indent=2)[:2000])

print("\n=== METADATA DIAGNOSTICS ===")

# for info in images[:MAX_IMAGES_TO_CHECK]:
#     filename = info["file_name"]
#     image_path = IMAGE_DIR / filename

#     print("\n" + "-" * 60)
#     print("COCO image ID:", info["id"])
#     print("Filename:", filename)
#     print("Expected path:", image_path)
#     print("File exists:", image_path.is_file())

#     # Check metadata directly in COCO JSON.
#     print("COCO metadata fields:")
#     for key in METADATA_CLASSES:
#         print(f"  {key}: {info.get(key, '<NOT FOUND>')}")

#     if not image_path.is_file():
#         # The image may be nested in a different directory.
#         matches = list(IMAGE_DIR.rglob(Path(filename).name))
#         print("Matching filenames found:", matches[:5])
#         continue

#     try:
#         with Image.open(image_path) as im:
#             exif = im.getexif()

#             print("Image format:", im.format)
#             print("EXIF tag count:", len(exif))
#             print("EXIF tag names:",
#                   [ExifTags.TAGS.get(k, k) for k in exif.keys()])

#             comment = exif.get(USER_COMMENT_TAG)
#             print("Raw UserComment:", repr(comment))

#             # Some metadata may be stored in the EXIF IFD.
#             try:
#                 exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
#                 print("EXIF IFD UserComment:",
#                       repr(exif_ifd.get(USER_COMMENT_TAG)))
#             except Exception as e:
#                 print("EXIF IFD read error:", repr(e))

#             if comment is not None:
#                 if isinstance(comment, bytes):
#                     if comment.startswith(b"ASCII\x00\x00\x00"):
#                         comment = comment[8:]
#                     elif comment.startswith(b"UNICODE\x00"):
#                         comment = comment[8:].decode(
#                             "utf-16-be", errors="replace"
#                         )
#                     else:
#                         comment = comment.decode(
#                             "utf-8", errors="replace"
#                         )

#                 print("Decoded UserComment:", repr(comment))

#                 try:
#                     parsed = json.loads(comment)
#                     print("Parsed metadata:", parsed)
#                     print("Metadata keys:", list(parsed.keys()))
#                 except (json.JSONDecodeError, TypeError) as e:
#                     print("JSON parsing failed:", repr(e))

#     except Exception as e:
#         print("Image read failed:", repr(e))

from pathlib import Path

IMAGE_DIR = Path("data_sub500_v2_metadata")



# ============================================================
# METADATA EXTRACTION
# ============================================================

def read_image_metadata(image_path):
    """Read JSON metadata from EXIF UserComment (tag 37510)."""
    try:
        with Image.open(image_path) as image:
            comment = image.getexif().get(37510)

            if comment is None:
                return {}

            # Normalize to bytes first.
            if isinstance(comment, str):
                comment = comment.encode("utf-8")

            # Remove EXIF UserComment encoding prefix.
            if comment.startswith(b"ASCII\x00\x00\x00"):
                comment = comment[8:]
                text = comment.decode("ascii")
            elif comment.startswith(b"UNICODE\x00"):
                comment = comment[8:]
                # Your writer may use UTF-16; try big-endian first.
                try:
                    text = comment.decode("utf-16-be")
                except UnicodeDecodeError:
                    text = comment.decode("utf-16-le")
            else:
                text = comment.decode("utf-8")

            metadata = json.loads(text)

            if not isinstance(metadata, dict):
                raise ValueError("Metadata JSON must be an object")

            return metadata

    except Exception as e:
        print(f"Metadata read error for {image_path}: {e}")
        return {}


for filename in ["000000087038.jpg", "000000006818.jpg"]:
    metadata = read_image_metadata(IMAGE_DIR / filename)
    print(filename, metadata)
    print("brightness =", metadata.get("brightness", "MISSING"))

def load_metadata(coco_gt):
    """
    Return a mapping: COCO image ID -> metadata dictionary.

    Metadata is read from the image's EXIF. If a field is
    missing there, values directly on the COCO image entry
    can serve as a fallback.
    """
    metadata_by_image = {}

    for image_info in coco_gt.dataset["images"]:
        image_id = image_info["id"]
        image_path = IMAGE_DIR / image_info["file_name"]

        metadata = {}

        if image_path.is_file():
            metadata = read_image_metadata(image_path)

        # EXIF values take precedence; use COCO image fields
        # for metadata keys that are not present in EXIF.
        for key in METADATA_CLASSES:
            if key not in metadata and key in image_info:
                metadata[key] = image_info[key]

        metadata_by_image[image_id] = metadata

    return metadata_by_image


# ============================================================
# COCO EVALUATION
# ============================================================

def get_coco_metrics(coco_gt, coco_dt, image_ids, iou_type="bbox"):
    """
    Evaluate a subset of image IDs using standard COCO metrics.

    Returns AP, AP50, AP75, AP-small, AP-medium, AP-large,
    AR@1, AR@10 and AR@100.
    """
    if not image_ids:
        return None

    evaluator = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    evaluator.params.imgIds = sorted(image_ids)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    stats = evaluator.stats

    return {
        "AP": stats[0],
        "AP50": stats[1],
        "AP75": stats[2],
        "AP_small": stats[3],
        "AP_medium": stats[4],
        "AP_large": stats[5],
        "AR@1": stats[6],
        "AR@10": stats[7],
        "AR@100": stats[8],
    }


def create_empty_coco_detections(coco_gt):
    """Create an empty COCO detection dataset."""
    coco_dt = COCO()
    coco_dt.dataset = {
        "images": coco_gt.dataset["images"],
        "categories": coco_gt.dataset["categories"],
        "annotations": [],
    }
    coco_dt.createIndex()
    return coco_dt


# ============================================================
# MAIN
# ============================================================

def main():
    # Load COCO ground truth.
    coco_gt = COCO(str(GT_JSON))

    with open(PREDICTIONS_JSON, "r") as f:
        predictions = json.load(f)

    # Ensure prediction image IDs match ground-truth image IDs.
    valid_image_ids = set(coco_gt.getImgIds())

    unknown_ids = {
        pred["image_id"]
        for pred in predictions
        if pred["image_id"] not in valid_image_ids
    }

    if unknown_ids:
        raise ValueError(
            f"Predictions contain unknown image IDs: "
            f"{sorted(unknown_ids)[:10]}"
        )

    # Load detections once. COCOeval will select the image
    # subset for each evaluation.
    if predictions:
        coco_dt = coco_gt.loadRes(predictions)
    else:
        coco_dt = create_empty_coco_detections(coco_gt)

    metadata_by_image = load_metadata(coco_gt)

    # Group image IDs by each metadata class and value.
    groups = defaultdict(list)

    for image_id, metadata in metadata_by_image.items():
        for metadata_class in METADATA_CLASSES:
            value = metadata.get(metadata_class, MISSING_VALUE)

            # These groups are intended to be categorical.
            # Reject continuous numeric metadata fields.
            if isinstance(value, (int, float)) and not isinstance(
                value, bool
            ):
                raise ValueError(
                    f"Metadata '{metadata_class}' has numeric "
                    f"value {value!r} for image {image_id}. "
                    "Use a categorical field instead."
                )

            groups[(metadata_class, str(value))].append(image_id)

    results = []

    # Evaluate each class-value combination.
    for (metadata_class, value), image_ids in sorted(groups.items()):
        print("\n" + "=" * 70)
        print(f"Metadata class: {metadata_class}")
        print(f"Value: {value}")
        print(f"Number of images: {len(image_ids)}")

        metrics = get_coco_metrics(
            coco_gt=coco_gt,
            coco_dt=coco_dt,
            image_ids=image_ids,
            iou_type=IOU_TYPE,
        )

        if metrics is None:
            continue

        # Count ground-truth objects in this subset.
        num_annotations = sum(
            len(coco_gt.getAnnIds(imgIds=[image_id]))
            for image_id in image_ids
        )

        row = {
            "metadata_class": metadata_class,
            "metadata_value": value,
            "num_images": len(image_ids),
            "num_gt_annotations": num_annotations,
            **{
                key: (
                    float(val)
                    if math.isfinite(float(val))
                    else None
                )
                for key, val in metrics.items()
            },
        }

        results.append(row)

        print(f"AP: {metrics['AP']:.4f}")
        print(f"AP50: {metrics['AP50']:.4f}")
        print(f"AP75: {metrics['AP75']:.4f}")

    # Export results to CSV.
    if results:
        fieldnames = list(results[0].keys())

        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\nSaved results to: {OUTPUT_CSV}")
    else:
        print("No metadata groups found to evaluate.")


if __name__ == "__main__":
    main()
