
import json
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

GT_FILE = "labels_sub500_v2.json"
PRED_FILE = "predictions_coco.json"
IMAGE_DIR = Path("data_sub500_v2")
IMAGE_ID = 87038

with open(GT_FILE) as f:
    gt = json.load(f)

with open(PRED_FILE) as f:
    preds = json.load(f)

# Find the GT image and its filename.
image_info = next(
    img for img in gt["images"]
    if img["id"] == IMAGE_ID
)
image_path = IMAGE_DIR / image_info["file_name"]
image = Image.open(image_path).convert("RGB")

categories = {
    cat["id"]: cat["name"]
    for cat in gt["categories"]
}

gt_annotations = [
    ann for ann in gt["annotations"]
    if ann["image_id"] == IMAGE_ID
]

image_preds = [
    p for p in preds
    if p["image_id"] == IMAGE_ID
]

draw = ImageDraw.Draw(image)

# Ground-truth boxes: green.
for ann in gt_annotations:
    x, y, w, h = ann["bbox"]
    draw.rectangle(
        [x, y, x + w, y + h],
        outline="lime",
        width=3,
    )
    label = categories.get(
        ann["category_id"], str(ann["category_id"])
    )
    draw.text((x, max(0, y - 14)), f"GT: {label}", fill="lime")

# Predicted boxes: red.
for pred in image_preds:
    x, y, w, h = pred["bbox"]
    draw.rectangle(
        [x, y, x + w, y + h],
        outline="red",
        width=2,
    )
    label = categories.get(
        pred["category_id"], str(pred["category_id"])
    )
    draw.text(
        (x, y + 2),
        f"{label} {pred['score']:.2f}",
        fill="red",
    )

print("Image:", image_info["file_name"])
print("Ground-truth objects:", len(gt_annotations))
print("Predictions:", len(image_preds))

plt.figure(figsize=(16, 10))
plt.imshow(image)
plt.axis("off")
plt.title(f"Image {IMAGE_ID}: GT (green), predictions (red)")
plt.show()




