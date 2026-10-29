"""
Reverse-engineer the BCCDP perturbation log from two COCO ground truths.

Given the clean ground truth (bccd_ground_truth.json) and the perturbed one
(bccdp_ground_truth.json, written by make_bccdp.py), this rebuilds a log in the same
format as bccdp_perturbation_log.json:

    <name>_missing.jpg  -> boxes present in <name>.jpg but absent from the variant
    <name>_wrong.jpg    -> boxes whose class differs between <name>.jpg and the variant

Every image in the perturbed GT is diffed against its source image in the clean GT, so
unsuffixed images that were edited in place are also reported (type inferred from the
diff). Boxes are matched by exact bbox, since make_bccdp.py never moves a box.

Known limits, inherent to working from COCO rather than the XMLs:
  - bbox_xyxy comes from the clamped COCO bbox, so a box that ran past the image edge in
    the XML will differ slightly from the original log.
  - Zero-area boxes are dropped by build_coco(), so they are not counted in n_original and
    a removed zero-area box cannot be detected.
  - The seed and fractions are not recoverable; observed fractions are reported instead.

Usage:
    python reverse_bccdp_log.py
    python reverse_bccdp_log.py --clean a.json --perturbed b.json -o log.json
    python reverse_bccdp_log.py --compare bccdp_perturbation_log.json
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

HERE = Path(__file__).resolve().parent
CLEAN_JSON = HERE / "bccd_ground_truth.json"
PERTURBED_JSON = HERE / "bccdp_ground_truth.json"
OUT_JSON = HERE / "bccdp_perturbation_log_reconstructed.json"

SUFFIXES = ("_missing", "_wrong")


# ============================================================
# LOADING
# ============================================================

def load_coco(path):
    """Return {file_name: [(class_name, bbox_xyxy), ...]} in annotation order."""
    coco = json.loads(Path(path).read_text())
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    file_of = {img["id"]: img["file_name"] for img in coco["images"]}
    boxes = {fn: [] for fn in file_of.values()}
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        xyxy = (int(x), int(y), int(x + w), int(y + h))
        boxes[file_of[ann["image_id"]]].append((cat_names[ann["category_id"]], xyxy))
    return boxes


def split_suffix(file_name):
    """'X_missing.jpg' -> ('X.jpg', 'missing'); 'X.jpg' -> ('X.jpg', None)."""
    p = Path(file_name)
    for suffix in SUFFIXES:
        if p.stem.endswith(suffix):
            return p.stem[: -len(suffix)] + p.suffix, suffix[1:]
    return file_name, None


# ============================================================
# DIFF
# ============================================================

def diff_boxes(clean, perturbed):
    """
    Match boxes between a source image and its variant.
    Returns (removed, changed, added), with removed/changed in clean annotation order.
    """
    # Exact (class, bbox) matches are untouched boxes.
    remaining = Counter(perturbed)
    unmatched_clean = []
    for box in clean:
        if remaining[box] > 0:
            remaining[box] -= 1
        else:
            unmatched_clean.append(box)

    # Left-over perturbed boxes, grouped by bbox, can explain a relabel.
    by_bbox = defaultdict(list)
    for (cls, bbox), n in remaining.items():
        by_bbox[bbox].extend([cls] * n)

    removed, changed = [], []
    for cls, bbox in unmatched_clean:
        if by_bbox[bbox]:
            changed.append({"from": cls, "to": by_bbox[bbox].pop(0), "bbox_xyxy": list(bbox)})
        else:
            removed.append({"class": cls, "bbox_xyxy": list(bbox)})

    added = [{"class": cls, "bbox_xyxy": list(bbox)}
             for bbox, classes in by_bbox.items() for cls in classes]
    return removed, changed, added


def build_entry(image, source, clean_boxes, removed, changed, added, declared):
    if declared:
        kind = declared
    elif removed and changed:
        kind = "mixed"
    else:
        kind = "missing" if removed else "wrong"

    entry = {"image": image, "source": source, "type": kind, "n_original": len(clean_boxes)}
    if removed or kind == "missing":
        entry["n_removed"] = len(removed)
        entry["removed"] = removed
    if changed or kind == "wrong":
        entry["n_changed"] = len(changed)
        entry["changed"] = changed
    if added:
        entry["n_added"] = len(added)
        entry["added"] = added
    return entry


def reconstruct(clean, perturbed):
    entries, warnings = [], []
    for image in sorted(perturbed):
        source, declared = split_suffix(image)
        if source not in clean:
            warnings.append(f"{image}: source {source} not in clean GT, skipped")
            continue

        removed, changed, added = diff_boxes(clean[source], perturbed[image])
        if not (removed or changed or added):
            if declared:
                warnings.append(f"{image}: identical to {source}, no perturbation found")
            continue

        if declared == "missing" and changed:
            warnings.append(f"{image}: _missing variant also has {len(changed)} relabelled box(es)")
        if declared == "wrong" and removed:
            warnings.append(f"{image}: _wrong variant also has {len(removed)} removed box(es)")
        if added:
            warnings.append(f"{image}: {len(added)} box(es) not present in {source}")
        if not declared:
            warnings.append(f"{image}: unsuffixed image differs from clean GT")

        entries.append(build_entry(image, source, clean[source], removed, changed, added, declared))

    # Same layout as make_bccdp.py: all _missing entries, then all _wrong ones.
    order = {"missing": 0, "wrong": 1}
    entries.sort(key=lambda e: (order.get(e["type"], 2), e["image"]))
    return entries, warnings


def observed_fraction(entries, kind, key):
    ratios = [e[key] / e["n_original"] for e in entries if e["type"] == kind and e["n_original"]]
    return round(sum(ratios) / len(ratios), 4) if ratios else None


# ============================================================
# COMPARE AGAINST AN EXISTING LOG
# ============================================================

def entry_signature(entry):
    removed = Counter((r["class"], tuple(r["bbox_xyxy"])) for r in entry.get("removed", []))
    changed = Counter((c["from"], c["to"], tuple(c["bbox_xyxy"])) for c in entry.get("changed", []))
    return entry["type"], entry["source"], removed, changed


def compare(reconstructed, reference_path):
    ref = {e["image"]: e for e in json.loads(Path(reference_path).read_text())["perturbations"]}
    rec = {e["image"]: e for e in reconstructed}

    matched, mismatched = 0, []
    for image in sorted(set(ref) & set(rec)):
        if entry_signature(ref[image]) == entry_signature(rec[image]):
            matched += 1
        else:
            mismatched.append(image)
    only_ref = sorted(set(ref) - set(rec))
    only_rec = sorted(set(rec) - set(ref))

    print(f"\nCompared with {Path(reference_path).name}: {matched}/{len(ref)} entries match exactly")
    for image in mismatched:
        r, c = entry_signature(ref[image]), entry_signature(rec[image])
        print(f"  MISMATCH {image}")
        print(f"    reference:     removed={dict(r[2])} changed={dict(r[3])}")
        print(f"    reconstructed: removed={dict(c[2])} changed={dict(c[3])}")
    for image in only_ref:
        print(f"  only in reference:     {image}")
    for image in only_rec:
        print(f"  only in reconstructed: {image}")


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--clean", type=Path, default=CLEAN_JSON, help="non-perturbed COCO ground truth")
    ap.add_argument("--perturbed", type=Path, default=PERTURBED_JSON, help="perturbed COCO ground truth")
    ap.add_argument("-o", "--out", type=Path, default=OUT_JSON, help="reconstructed log to write")
    ap.add_argument("--compare", type=Path, help="existing perturbation log to check the result against")
    args = ap.parse_args()

    clean = load_coco(args.clean)
    perturbed = load_coco(args.perturbed)
    entries, warnings = reconstruct(clean, perturbed)

    args.out.write_text(json.dumps({
        "seed": None,
        "missing_fraction": observed_fraction(entries, "missing", "n_removed"),
        "wrong_fraction": observed_fraction(entries, "wrong", "n_changed"),
        "reconstructed_from": {"clean": args.clean.name, "perturbed": args.perturbed.name},
        "perturbations": entries,
    }, indent=2))

    counts = Counter(e["type"] for e in entries)
    n_removed = sum(e.get("n_removed", 0) for e in entries)
    n_changed = sum(e.get("n_changed", 0) for e in entries)
    print(f"Found {counts['missing']} _missing images ({n_removed} boxes removed in total) and "
          f"{counts['wrong']} _wrong images ({n_changed} boxes relabelled in total)"
          + (f", plus {counts['mixed']} mixed" if counts["mixed"] else ""))
    print(f"Reconstructed log -> {args.out.name}")
    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for w in warnings:
            print(f"  {w}")

    if args.compare:
        compare(entries, args.compare)


if __name__ == "__main__":
    main()
