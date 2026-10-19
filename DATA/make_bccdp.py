"""
Build the perturbed BCCD dataset (BCCDP) for label-noise auditing.

Picks 50 random original images from bccdp_images/ and adds duplicates of them:
    25 x <name>_missing.jpg  - annotation has a fraction of its boxes deleted
    25 x <name>_wrong.jpg    - annotation has a fraction of its boxes relabelled to a different class

The matching <name>_missing.xml / <name>_wrong.xml are written to bccdp_annotations/.
The original files are left untouched. Then bccdp_ground_truth.json (COCO format) is
built from every XML in bccdp_annotations/, and bccdp_perturbation_log.json records
exactly which boxes were removed / relabelled.

Re-running is safe: any existing *_missing / *_wrong files are deleted first, and the
fixed seed reproduces the same selection.
"""

import json
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

HERE = Path(__file__).resolve().parent
IMAGE_DIR = HERE / "bccdp_images"
ANN_DIR = HERE / "bccdp_annotations"
GT_JSON = HERE / "bccdp_ground_truth.json"
LOG_JSON = HERE / "bccdp_perturbation_log.json"

SEED = 42
N_MISSING = 25
N_WRONG = 25
MISSING_FRACTION = 0.3   # share of boxes deleted in each _missing annotation
WRONG_FRACTION = 0.3     # share of boxes relabelled in each _wrong annotation

CATEGORIES = {"RBC": 1, "WBC": 2, "Platelets": 3}
SUFFIXES = ("_missing", "_wrong")


# ============================================================
# PERTURBATION
# ============================================================

def n_to_perturb(n_objects, fraction):
    """At least one box, rounded from the fraction."""
    return min(n_objects, max(1, round(n_objects * fraction)))


def box_of(obj):
    b = obj.find("bndbox")
    return [int(float(b.find(k).text)) for k in ("xmin", "ymin", "xmax", "ymax")]


def write_variant(stem, suffix, tree):
    new_stem = stem + suffix
    root = tree.getroot()
    root.find("filename").text = new_stem + ".jpg"
    path_el = root.find("path")
    if path_el is not None:
        path_el.text = str(Path(path_el.text).with_name(new_stem + ".jpg"))
    tree.write(ANN_DIR / f"{new_stem}.xml", encoding="utf-8")
    shutil.copy2(IMAGE_DIR / f"{stem}.jpg", IMAGE_DIR / f"{new_stem}.jpg")
    return new_stem


def make_missing(stem, rng):
    tree = ET.parse(ANN_DIR / f"{stem}.xml")
    root = tree.getroot()
    objs = root.findall("object")
    removed = rng.sample(objs, n_to_perturb(len(objs), MISSING_FRACTION))
    for obj in removed:
        root.remove(obj)
    new_stem = write_variant(stem, "_missing", tree)
    return {
        "image": new_stem + ".jpg", "source": stem + ".jpg", "type": "missing",
        "n_original": len(objs), "n_removed": len(removed),
        "removed": [{"class": o.find("name").text.strip(), "bbox_xyxy": box_of(o)} for o in removed],
    }


def make_wrong(stem, rng):
    tree = ET.parse(ANN_DIR / f"{stem}.xml")
    objs = tree.getroot().findall("object")
    changed = []
    for obj in rng.sample(objs, n_to_perturb(len(objs), WRONG_FRACTION)):
        name_el = obj.find("name")
        old = name_el.text.strip()
        new = rng.choice([c for c in CATEGORIES if c != old])
        name_el.text = new
        changed.append({"from": old, "to": new, "bbox_xyxy": box_of(obj)})
    new_stem = write_variant(stem, "_wrong", tree)
    return {
        "image": new_stem + ".jpg", "source": stem + ".jpg", "type": "wrong",
        "n_original": len(objs), "n_changed": len(changed), "changed": changed,
    }


# ============================================================
# COCO GROUND TRUTH
# ============================================================

def build_coco(ann_dir, image_dir):
    images, annotations = [], []
    skipped = 0
    for img_id, xml_path in enumerate(sorted(ann_dir.glob("*.xml")), start=1):
        root = ET.parse(xml_path).getroot()
        file_name = xml_path.stem + ".jpg"
        if not (image_dir / file_name).exists():
            raise FileNotFoundError(f"No image for {xml_path.name}")
        W = int(root.find("size/width").text)
        H = int(root.find("size/height").text)
        images.append({"id": img_id, "file_name": file_name, "width": W, "height": H})
        for obj in root.findall("object"):
            x1, y1, x2, y2 = box_of(obj)
            x1, x2 = max(0, min(x1, W)), max(0, min(x2, W))
            y1, y2 = max(0, min(y1, H)), max(0, min(y2, H))
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                skipped += 1
                continue
            annotations.append({
                "id": len(annotations) + 1, "image_id": img_id,
                "category_id": CATEGORIES[obj.find("name").text.strip()],
                "bbox": [x1, y1, w, h], "area": w * h, "iscrowd": 0,
            })
    coco = {
        "info": {"description": "BCCDP (perturbed BCCD) - converted from Pascal VOC", "version": "1.0"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": n, "supercategory": "cell"} for n, i in CATEGORIES.items()],
    }
    return coco, skipped


# ============================================================
# MAIN
# ============================================================

def main():
    # Clear any previous perturbed variants so the run is reproducible.
    for d, ext in ((IMAGE_DIR, "jpg"), (ANN_DIR, "xml")):
        for suffix in SUFFIXES:
            for f in d.glob(f"*{suffix}.{ext}"):
                f.unlink()

    stems = sorted(p.stem for p in ANN_DIR.glob("*.xml"))
    missing_imgs = [s for s in stems if not (IMAGE_DIR / f"{s}.jpg").exists()]
    if missing_imgs:
        raise FileNotFoundError(f"{len(missing_imgs)} annotations have no image, e.g. {missing_imgs[:3]}")

    rng = random.Random(SEED)
    chosen = rng.sample(stems, N_MISSING + N_WRONG)
    log = [make_missing(s, rng) for s in chosen[:N_MISSING]]
    log += [make_wrong(s, rng) for s in chosen[N_MISSING:]]

    LOG_JSON.write_text(json.dumps({
        "seed": SEED, "missing_fraction": MISSING_FRACTION, "wrong_fraction": WRONG_FRACTION,
        "perturbations": log,
    }, indent=2))

    coco, skipped = build_coco(ANN_DIR, IMAGE_DIR)
    GT_JSON.write_text(json.dumps(coco))

    n_removed = sum(e.get("n_removed", 0) for e in log)
    n_changed = sum(e.get("n_changed", 0) for e in log)
    print(f"Added {N_MISSING} _missing images ({n_removed} boxes removed) and "
          f"{N_WRONG} _wrong images ({n_changed} boxes relabelled)")
    print(f"{GT_JSON.name}: {len(coco['images'])} images, {len(coco['annotations'])} boxes "
          f"({skipped} zero-area boxes skipped)")
    print(f"Perturbation details -> {LOG_JSON.name}")


if __name__ == "__main__":
    main()
