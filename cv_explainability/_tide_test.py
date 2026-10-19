
from tidecv import TIDE, datasets
from pprint import pprint 
tide = TIDE()
tide.evaluate(
    datasets.COCO("labels_sub500_v2.json"),
    datasets.COCOResult("predictions_coco.json"),
    mode=TIDE.BOX,
)

tide.summarize()
# tide.plot()


# Inspect the error information available in your installed TIDE version.
print("Runs:", list(tide.runs.keys()))

run = next(iter(tide.runs.values()))

print("Run attributes:", vars(run).keys())
print("Number of error records:", len(run.errors))

for error in run.errors[:5]:
    print("\nError type:", type(error).__name__)
    pprint(vars(error))

run = tide.runs["predictions_coco"]

background_errors = [
    e for e in run.errors
    if type(e).__name__ == "BackgroundError"
    and e.pred is not None
]

background_errors.sort(
    key=lambda e: e.pred["score"],
    reverse=True,
)

for e in background_errors[:20]:
    p = e.pred
    print(
        f'image={p["image"]}, '
        f'class={p["class"]}, '
        f'score={p["score"]:.3f}, '
        f'bbox={p["bbox"]}'
    )



import json
from pathlib import Path
from collections import Counter

import pandas as pd

# ---- Configuration ----
GT_FILE = "labels_sub500_v2.json"
RUN_NAME = "predictions_coco"
OUTPUT_DIR = Path("tide_diagnostics")
OUTPUT_DIR.mkdir(exist_ok=True)

# Existing TIDE object, created earlier:
run = tide.runs[RUN_NAME]

with open(GT_FILE, "r") as f:
    gt_data = json.load(f)

class_names = {
    int(c["id"]): c["name"]
    for c in gt_data["categories"]
}

# Ground-truth object count per class
gt_counts = Counter(
    int(a["category_id"])
    for a in gt_data["annotations"]
    if not a.get("iscrowd", 0)
)


import pandas as pd

def class_name(class_id):
    if pd.isna(class_id):
        return "unknown"

    class_id = int(class_id)
    return class_names.get(class_id, f"class_{class_id}")

def get_id(obj):
    """Internal TIDE object ID, used to avoid counting the
    same object repeatedly within an error category."""
    if obj is None:
        return None
    return obj.get("_id")

# ---- Extract TIDE error records ----
records = []
seen = set()

for err in run.errors:
    error_type = type(err).__name__
    pred = getattr(err, "pred", None)
    gt = getattr(err, "gt", None)

    # Deduplicate repeated records for the same object/type.
    # Keep separate error categories separate.
    if pred is not None:
        object_id = get_id(pred)
        identity = (
            error_type,
            "pred",
            object_id if object_id is not None else id(pred)
        )
    elif gt is not None:
        object_id = get_id(gt)
        identity = (
            error_type,
            "gt",
            object_id if object_id is not None else id(gt)
        )
    else:
        continue

    if identity in seen:
        continue
    seen.add(identity)

    pred_class = (
        int(pred["class"]) if pred is not None else None
    )
    gt_class = (
        int(gt["class"]) if gt is not None else None
    )

    records.append({
        "error_type": error_type,
        "image_id": (
            pred.get("image") if pred is not None
            else gt.get("image")
        ),
        "pred_id": get_id(pred),
        "gt_id": get_id(gt),
        "pred_class_id": pred_class,
        "gt_class_id": gt_class,
        "pred_class": class_name(pred_class),
        "gt_class": class_name(gt_class),
        "score": (
            float(pred["score"]) if pred is not None
            else None
        ),
        "iou": (
            pred.get("info", {}).get("iou")
            if pred is not None else None
        ),
        "bbox": pred.get("bbox") if pred is not None else None,
    })

errors = pd.DataFrame(records)

if errors.empty:
    raise RuntimeError(
        "No error records found. Check the TIDE run and evaluation."
    )

errors.to_csv(
    OUTPUT_DIR / "all_error_records.csv", index=False
)

print(f"Unique object/error records: {len(errors)}")
print(f"Images represented: {errors['image_id'].nunique()}")
print("\nError types found:")
print(errors["error_type"].value_counts().to_string())



# Map internal TIDE names to user-friendly labels.
TYPE_LABELS = {
    "ClassError": "Classification",
    "BoxError": "Localization",
    "OtherError": "Classification + localization",
    "DuplicateError": "Duplicate",
    "BackgroundError": "Background false positive",
    "MissedError": "Missed ground truth",
}

errors["error_label"] = errors["error_type"].map(
    lambda x: TYPE_LABELS.get(x, x)
)

# Overall error counts
overall = (
    errors.groupby("error_label")
    .size()
    .rename("count")
    .sort_values(ascending=False)
    .reset_index()
)

print("\n=== Error counts by type ===")
print(overall.to_string(index=False))
overall.to_csv(OUTPUT_DIR / "error_type_counts.csv", index=False)

# Count errors by class.
# Classification errors are attributed to the TRUE class for
# the per-class failure table. Background errors are attributed
# to the PREDICTED class.
def relevant_class(row):
    if row["error_type"] == "BackgroundError":
        return row["pred_class_id"]
    if row["gt_class_id"] is not None:
        return row["gt_class_id"]
    return row["pred_class_id"]

errors["diagnostic_class_id"] = errors.apply(
    relevant_class, axis=1
)
print("idk what im looking at btw")
print(
    errors["diagnostic_class_id"]
    .isna()
    .sum()
)


errors["diagnostic_class"] = errors[
    "diagnostic_class_id"
].map(class_name)

class_error_counts = pd.crosstab(
    errors["diagnostic_class"],
    errors["error_label"]
)

# Include ground-truth frequency as context.
class_error_counts["ground_truth_objects"] = (
    class_error_counts.index.map({
        class_name(k): v for k, v in gt_counts.items()
    })
)

# Rates use number of ground-truth objects as the denominator.
# For background false positives this is a contextual rate,
# not a formal false-positive rate.
for col in class_error_counts.columns:
    if col == "ground_truth_objects":
        continue
    class_error_counts[col + "_per_100_gt"] = (
        100 * class_error_counts[col]
        / class_error_counts["ground_truth_objects"].replace(0, pd.NA)
    )

class_error_counts = class_error_counts.fillna(0)
class_error_counts.to_csv(
    OUTPUT_DIR / "per_class_errors.csv"
)

print("\n=== Per-class error table ===")
print(class_error_counts.to_string())



# TIDE ClassError contains both the incorrect prediction
# and the associated ground-truth object.
cls = errors[errors["error_type"] == "ClassError"].copy()

if not cls.empty:
    confusion = (
        cls.groupby(["gt_class", "pred_class"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False)
    )

    print("\n=== Most common classification confusions ===")
    print(confusion.head(30).to_string(index=False))

    confusion.to_csv(
        OUTPUT_DIR / "class_confusions.csv", index=False
    )

    # Show the most frequent wrong predicted class for each
    # ground-truth class.
    top_confusion = (
        confusion.sort_values("count", ascending=False)
        .drop_duplicates("gt_class")
    )

    print("\n=== Top confusion for each true class ===")
    print(top_confusion.to_string(index=False))
else:
    print("No TIDE ClassError records found.")



print("\n=== Diagnostic findings ===")

# 1. Which error categories have the most records?
for _, row in overall.head(3).iterrows():
    print(
        f"- {row['error_label']}: {int(row['count'])} "
        "deduplicated error records. Investigate this category."
    )

# 2. Classes with the most missed ground-truth objects
misses = errors[errors["error_type"] == "MissedError"]

if not misses.empty:
    top_missed = (
        misses.groupby("gt_class")
        .size()
        .sort_values(ascending=False)
        .head(5)
    )
    print("\nClasses with the most missed objects:")
    for name, count in top_missed.items():
        print(f"- {name}: {count} missed-object records")

# 3. Most common class confusions
if not cls.empty:
    print("\nMost frequent class confusions:")
    for _, row in confusion.head(5).iterrows():
        print(
            f"- True {row['gt_class']} predicted as "
            f"{row['pred_class']}: {int(row['count'])} records"
        )

# 4. Classes with the most localization errors
loc = errors[errors["error_type"] == "BoxError"]

if not loc.empty:
    top_loc = (
        loc.groupby("gt_class")
        .size()
        .sort_values(ascending=False)
        .head(5)
    )
    print("\nClasses with the most localization-error records:")
    for name, count in top_loc.items():
        print(f"- {name}: {count} localization-error records")

# 5. Predicted classes most associated with background errors
bkg = errors[errors["error_type"] == "BackgroundError"]

if not bkg.empty:
    top_bkg = (
        bkg.groupby("pred_class")
        .size()
        .sort_values(ascending=False)
        .head(5)
    )
    print("\nPredicted classes most associated with background errors:")
    for name, count in top_bkg.items():
        print(f"- {name}: {count} background-error records")

print(f"\nCSV reports saved to: {OUTPUT_DIR.resolve()}")
