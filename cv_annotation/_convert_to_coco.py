import json

input_file = "prediction_jsons/predictions_yolo11x.json"
output_file = "predictions_coco.json"

with open(input_file, "r") as f:
    data = json.load(f)

coco_predictions = []

for item in data:
    # COCO image_id should match the GT annotation's image_id.
    # Convert numeric-looking strings to integers.
    image_id = item["image_id"]

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

with open(output_file, "w") as f:
    json.dump(coco_predictions, f)

print(f"Converted {len(coco_predictions)} detections")
print(f"Saved to {output_file}")
