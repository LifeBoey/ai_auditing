"""Load COCO ground truth and predictions in either supported format."""

import json
from collections import defaultdict


def load_gt(path):
    gt = json.load(open(path, encoding="utf-8"))
    anns = defaultdict(list)
    for a in gt["annotations"]:
        anns[a["image_id"]].append(a)
    return gt, anns


def load_preds(path, gt):
    """
    -> {image_id: [{"category_id", "bbox" (xywh), "score"}]}

    Accepts
      * flat COCO results: [{"image_id", "category_id", "bbox": [x, y, w, h], "score"}]
      * per-image lists:   [{"file_name", "prediction": [{"class_name"/"class_id", "bbox": [x1, y1, x2, y2], "score"}]}]
        (matched to GT by file_name, classes by class_name, falling back to class_id)
    """
    raw = json.load(open(path, encoding="utf-8"))
    fid = {im["file_name"]: im["id"] for im in gt["images"]}
    ids = set(fid.values())
    cid = {c["name"]: c["id"] for c in gt["categories"]}
    out, skipped_img, skipped_cls = defaultdict(list), 0, 0
    for r in raw:
        if "prediction" in r:
            iid = fid.get(r.get("file_name"))
            if iid is None:
                skipped_img += 1
                continue
            for p in r["prediction"]:
                c = cid.get(p.get("class_name"), p.get("class_id"))
                if c not in cid.values():
                    skipped_cls += 1
                    continue
                x1, y1, x2, y2 = p["bbox"]
                out[iid].append(dict(category_id=c, bbox=[x1, y1, x2 - x1, y2 - y1], score=float(p["score"])))
        else:
            if r["image_id"] not in ids:
                skipped_img += 1
                continue
            out[r["image_id"]].append(dict(category_id=r["category_id"], bbox=list(r["bbox"]),
                                           score=float(r["score"])))
    if skipped_img:
        print(f"[warn] {skipped_img} prediction entries refer to images not in the GT; skipped")
    if skipped_cls:
        print(f"[warn] {skipped_cls} predicted boxes have a class not in the GT; skipped")
    return out
