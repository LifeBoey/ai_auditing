"""
Build a perturbed subset of any COCO-format detection dataset for label-noise auditing.

Generic version of make_bccdp.py: instead of Pascal VOC XMLs it reads a clean COCO ground
truth plus its image directory (M images), samples N of them (N < M), and writes:

    <out-dir>/                      N images; the perturbed ones are renamed
        <name>_missing.<ext>        - annotation has a fraction of its boxes deleted
        <name>_wrong.<ext>          - annotation has a fraction of its boxes relabelled
        <name>.<ext>                - every other sampled image, unchanged
    <out-dir>_ground_truth.json     COCO ground truth for exactly those images
    <out-dir>_perturbation_log.json which boxes were removed / relabelled (make_bccdp format)

With --keep-originals the clean copy of each perturbed image is kept as well (as
make_bccdp.py does), so the directory holds N + n_missing + n_wrong images.

Only images that exist in the image directory are sampled. Images are only perturbed if they
have at least one box. The fixed seed reproduces the same selection.

Usage:
    python _make_perturbed_subset.py --gt clean.json --images imgs/ -n 200
    python _make_perturbed_subset.py --gt clean.json --images imgs/ -n 200 \\
        --n-missing 30 --n-wrong 30 --missing-fraction 0.5 --out-dir subset/ --overwrite
"""

import argparse
import copy
import json
import random
import shutil
from pathlib import Path


# ============================================================
# DEFAULTS
# ============================================================

SEED = 42
N_IMAGES = 100
PERTURB_SHARE = 0.25     # default n_missing / n_wrong, each as a share of N
MISSING_FRACTION = 0.3   # share of boxes deleted in each _missing annotation
WRONG_FRACTION = 0.3     # share of boxes relabelled in each _wrong annotation


# ============================================================
# PERTURBATION
# ============================================================

def n_to_perturb(n_objects, fraction):
    """At least one box, rounded from the fraction."""
    return min(n_objects, max(1, round(n_objects * fraction)))


def xyxy(ann):
    x, y, w, h = ann["bbox"]
    return [x, y, x + w, y + h]


def variant_name(file_name, suffix):
    p = Path(file_name)
    return str(p.with_name(p.stem + suffix + p.suffix).as_posix())


def make_missing(img, anns, cat_names, fraction, rng):
    removed = rng.sample(anns, n_to_perturb(len(anns), fraction))
    removed_ids = {id(a) for a in removed}
    kept = [a for a in anns if id(a) not in removed_ids]
    new_name = variant_name(img["file_name"], "_missing")
    entry = {
        "image": new_name, "source": img["file_name"], "type": "missing",
        "n_original": len(anns), "n_removed": len(removed),
        "removed": [{"class": cat_names[a["category_id"]], "bbox_xyxy": xyxy(a)} for a in removed],
    }
    return new_name, kept, entry


def make_wrong(img, anns, cat_names, fraction, rng):
    anns = [copy.deepcopy(a) for a in anns]
    changed = []
    for ann in rng.sample(anns, n_to_perturb(len(anns), fraction)):
        old = ann["category_id"]
        new = rng.choice([c for c in cat_names if c != old])
        ann["category_id"] = new
        changed.append({"from": cat_names[old], "to": cat_names[new], "bbox_xyxy": xyxy(ann)})
    new_name = variant_name(img["file_name"], "_wrong")
    entry = {
        "image": new_name, "source": img["file_name"], "type": "wrong",
        "n_original": len(anns), "n_changed": len(changed), "changed": changed,
    }
    return new_name, anns, entry


# ============================================================
# MAIN
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--gt", type=Path, required=True, help="clean COCO ground truth json")
    ap.add_argument("--images", type=Path, required=True, help="directory of the M clean images")
    ap.add_argument("-n", "--n-images", type=int, default=N_IMAGES, help="size N of the subset")
    ap.add_argument("--n-missing", type=int, help=f"images given missing boxes (default {PERTURB_SHARE:.0%} of N)")
    ap.add_argument("--n-wrong", type=int, help=f"images given wrong labels (default {PERTURB_SHARE:.0%} of N)")
    ap.add_argument("--missing-fraction", type=float, default=MISSING_FRACTION,
                    help="share of boxes deleted per _missing image")
    ap.add_argument("--wrong-fraction", type=float, default=WRONG_FRACTION,
                    help="share of boxes relabelled per _wrong image")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--keep-originals", action="store_true",
                    help="also keep the clean copy of every perturbed image")
    ap.add_argument("--out-dir", type=Path, help="output image directory (default <images>_perturbed)")
    ap.add_argument("--out-gt", type=Path, help="output ground truth (default <out-dir>_ground_truth.json)")
    ap.add_argument("--out-log", type=Path, help="output log (default <out-dir>_perturbation_log.json)")
    ap.add_argument("--overwrite", action="store_true", help="clear a non-empty output directory first")
    args = ap.parse_args()

    n = args.n_images
    if args.n_missing is None:
        args.n_missing = round(n * PERTURB_SHARE)
    if args.n_wrong is None:
        args.n_wrong = round(n * PERTURB_SHARE)
    if args.n_missing + args.n_wrong > n:
        ap.error(f"n_missing + n_wrong ({args.n_missing + args.n_wrong}) exceeds N ({n})")
    for name in ("missing_fraction", "wrong_fraction"):
        if not 0 < getattr(args, name) <= 1:
            ap.error(f"--{name.replace('_', '-')} must be in (0, 1]")

    args.out_dir = args.out_dir or args.images.with_name(args.images.name + "_perturbed")
    args.out_gt = args.out_gt or args.out_dir.with_name(args.out_dir.name + "_ground_truth.json")
    args.out_log = args.out_log or args.out_dir.with_name(args.out_dir.name + "_perturbation_log.json")
    return args


def main():
    args = parse_args()

    coco = json.loads(args.gt.read_text())
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    if len(cat_names) < 2 and args.n_wrong:
        raise ValueError("Need at least 2 categories to relabel boxes")
    anns_of = {img["id"]: [] for img in coco["images"]}
    for ann in coco["annotations"]:
        anns_of[ann["image_id"]].append(ann)

    # Only sample images that are actually on disk.
    on_disk = [img for img in coco["images"] if (args.images / img["file_name"]).is_file()]
    n_absent = len(coco["images"]) - len(on_disk)
    if args.n_images >= len(on_disk):
        raise ValueError(f"N ({args.n_images}) must be smaller than the {len(on_disk)} images found on disk")

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{args.out_dir} is not empty (pass --overwrite to clear it)")
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    subset = sorted(rng.sample(on_disk, args.n_images), key=lambda img: img["file_name"])
    perturbable = [img for img in subset if anns_of[img["id"]]]
    n_perturb = args.n_missing + args.n_wrong
    if n_perturb > len(perturbable):
        raise ValueError(f"Only {len(perturbable)} sampled images have boxes, need {n_perturb}")
    chosen = rng.sample(perturbable, n_perturb)
    plan = {img["id"]: "missing" for img in chosen[:args.n_missing]}
    plan.update({img["id"]: "wrong" for img in chosen[args.n_missing:]})

    # (file_name, source image record, annotations) for every image that goes in the output.
    outputs, log = [], []
    for img in chosen:
        make = make_missing if plan[img["id"]] == "missing" else make_wrong
        fraction = args.missing_fraction if plan[img["id"]] == "missing" else args.wrong_fraction
        new_name, anns, entry = make(img, anns_of[img["id"]], cat_names, fraction, rng)
        outputs.append((new_name, img, anns))
        log.append(entry)
    for img in subset:
        if img["id"] not in plan or args.keep_originals:
            outputs.append((img["file_name"], img, anns_of[img["id"]]))
    outputs.sort(key=lambda o: o[0])

    images, annotations = [], []
    for img_id, (file_name, src, anns) in enumerate(outputs, start=1):
        dst = args.out_dir / file_name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.images / src["file_name"], dst)
        images.append({**src, "id": img_id, "file_name": file_name})
        for ann in anns:
            annotations.append({**ann, "id": len(annotations) + 1, "image_id": img_id})

    out_coco = {
        "info": {**coco.get("info", {}),
                 "description": f"Perturbed subset of {args.gt.name} for label-noise auditing"},
        "licenses": coco.get("licenses", []),
        "images": images,
        "annotations": annotations,
        "categories": coco["categories"],
    }
    args.out_gt.write_text(json.dumps(out_coco))
    args.out_log.write_text(json.dumps({
        "seed": args.seed, "missing_fraction": args.missing_fraction, "wrong_fraction": args.wrong_fraction,
        "source_ground_truth": args.gt.name, "n_images": args.n_images,
        "keep_originals": args.keep_originals,
        "perturbations": log,
    }, indent=2))

    n_removed = sum(e.get("n_removed", 0) for e in log)
    n_changed = sum(e.get("n_changed", 0) for e in log)
    if n_absent:
        print(f"Note: {n_absent} images in {args.gt.name} were not found in {args.images} and were ignored")
    print(f"Sampled {args.n_images} of {len(on_disk)} images (seed {args.seed})")
    print(f"Made {args.n_missing} _missing images ({n_removed} boxes removed) and "
          f"{args.n_wrong} _wrong images ({n_changed} boxes relabelled)")
    print(f"{args.out_dir}: {len(images)} images")
    print(f"{args.out_gt.name}: {len(images)} images, {len(annotations)} boxes")
    print(f"Perturbation details -> {args.out_log.name}")


if __name__ == "__main__":
    main()
