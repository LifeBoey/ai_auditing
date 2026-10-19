
# import json
# from collections import Counter

# GT_FILE = "labels_sub500_v2.json"
# PRED_FILE = "predictions_coco.json"

# with open(GT_FILE) as f:
#     gt = json.load(f)

# with open(PRED_FILE) as f:
#     preds = json.load(f)

# gt_image_ids = {img["id"] for img in gt["images"]}
# gt_cat_ids = {cat["id"] for cat in gt["categories"]}
# gt_annot_image_ids = {ann["image_id"] for ann in gt["annotations"]}
# gt_annot_cat_ids = {ann["category_id"] for ann in gt["annotations"]}


# for image in gt['images']:
#      if '_' in image['file_name']:
#          print(image)
#          break

# id_map = {
#     img["id"]: i
#     for i, img in enumerate(gt["images"], start=1)
# }

# for img in gt["images"]:
#     img["id"] = id_map[img["id"]]

# for ann in gt["annotations"]:
#     ann["image_id"] = id_map[ann["image_id"]]


# # Map filenames to the integer image IDs in the GT file.
# filename_to_id = {
#     img["file_name"]: img["id"]
#     for img in gt["images"]
# }

# # Your prediction records contain image_id as a filename stem.
# # Convert "000000000139" -> "000000000139.jpg" -> GT integer ID.
# for pred in preds:
#     pred["image_id"] = filename_to_id[pred["file_name"]]
#     original_id = pred["image_id"]
#     filename = f"{original_id}.jpg"

#     if filename not in filename_to_id:
#         raise ValueError(f"No GT image found for {filename}")

#     pred["image_id"] = filename_to_id[filename]

# print("Mapped all prediction image IDs.")


# pred_image_ids = {p["image_id"] for p in preds}
# pred_cat_ids = {p["category_id"] for p in preds}

# print("GT images:", len(gt_image_ids))
# print("GT annotations:", len(gt["annotations"]))
# print("GT categories:", len(gt_cat_ids))
# print("Predictions:", len(preds))
# print("GT[categories]:", gt["categories"])


# print("GT image ID types:",
#       Counter(type(x).__name__ for x in gt_image_ids))
# print("Prediction image ID types:",
#       Counter(type(x).__name__ for x in pred_image_ids))

# print("Prediction image IDs absent from GT:",
#       len(pred_image_ids - gt_image_ids))
# print("GT image IDs with no predictions:",
#       len(gt_image_ids - pred_image_ids))

# gt_ids = {img["id"] for img in gt["images"]}
# pred_ids = {p["image_id"] for p in preds}

# print("Unmatched prediction IDs:", len(pred_ids - gt_ids))
# print("GT images with no predictions:", len(gt_ids - pred_ids))

# print("Prediction category IDs absent from GT:",
#       pred_cat_ids - gt_cat_ids)
# print("GT categories with annotations:",
#       len(gt_annot_cat_ids))
# print("Prediction categories:",
#       sorted(pred_cat_ids))

# bad_boxes = [
#     p for p in preds
#     if len(p["bbox"]) != 4
#     or p["bbox"][2] <= 0
#     or p["bbox"][3] <= 0
# ]
# print("Invalid COCO-format boxes:", len(bad_boxes))


import json

GT_FILE = "labels_sub500_v2.json"
INPUT_FILE = "prediction_jsons/predictions_yolo11x.json"
OUTPUT_FILE = "predictions_coco.json"

with open(GT_FILE) as f:
    gt = json.load(f)

with open(INPUT_FILE) as f:
    data = json.load(f)

# Map each exact GT filename to its integer COCO image ID.
filename_to_id = {
    img["file_name"]: img["id"]
    for img in gt["images"]
}

coco_predictions = []

for item in data:
    filename = item["file_name"]

    if filename not in filename_to_id:
        raise ValueError(
            f"Prediction image {filename!r} not found in GT images"
        )

    image_id = filename_to_id[filename]

    for pred in item["prediction"]:
        x1, y1, x2, y2 = pred["bbox"]

        coco_predictions.append({
            "image_id": image_id,
            "category_id": int(pred["class_id"]),
            "bbox": [
                float(x1),
                float(y1),
                float(x2 - x1),
                float(y2 - y1),
            ],
            "score": float(pred["score"]),
        })

with open(OUTPUT_FILE, "w") as f:
    json.dump(coco_predictions, f)

print(f"Converted {len(coco_predictions)} detections")
print(f"Saved to {OUTPUT_FILE}")


mapped_ids = {
    p["image_id"] for p in coco_predictions
}
gt_ids = {img["id"] for img in gt["images"]}

print("Prediction image IDs absent from GT:", mapped_ids - gt_ids)
print("GT images without predictions:", gt_ids - mapped_ids)

