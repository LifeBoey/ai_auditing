"""
check_against_pycocotools.py -- verify metrics.py against pycocotools on every group in tags.csv,
and that each aspect's groups add up to the overall TP / FN counts.

    python check_against_pycocotools.py --preds ... --labels ... --tags aspects_out/tags.csv
"""

import argparse
import contextlib
import io

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import metrics as M
from io_utils import load_gt, load_preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--tags", required=True)
    args = ap.parse_args()

    gt, anns = load_gt(args.labels)
    preds = load_preds(args.preds, gt)
    fid = {im["file_name"]: im["id"] for im in gt["images"]}
    classes = {c["id"]: c["name"] for c in gt["categories"]}
    flat = [dict(image_id=i, **d) for i, ds in preds.items() for d in ds]
    with contextlib.redirect_stdout(io.StringIO()):
        G = COCO(args.labels)
        D = G.loadRes(flat)
    evals = {i: M.evaluate_image(anns.get(i, []), preds.get(i, [])) for i in sorted(fid.values())}

    tags = pd.read_csv(args.tags, dtype=str, keep_default_na=False)
    tags = tags[tags["file_name"].isin(fid)]
    ids_all = sorted(fid[f] for f in tags["file_name"])
    conf, _ = M.best_f1_conf(M.pool([evals[i] for i in ids_all]))
    tot = M.counts_at(M.pool([evals[i] for i in ids_all]), conf)
    worst, n, bad_sums = 0.0, 0, []
    for col in [c for c in tags.columns if c.endswith("__group")]:
        tp_sum = fn_sum = 0
        for g in sorted(tags[col].unique()):
            ids = sorted(fid[f] for f in tags.loc[tags[col] == g, "file_name"])
            with contextlib.redirect_stdout(io.StringIO()):
                E = COCOeval(G, D, "bbox")
                E.params.imgIds = ids
                E.evaluate(); E.accumulate(); E.summarize()
            s = M.summary(M.pool([evals[i] for i in ids]), classes)
            d = max(abs(E.stats[0] - s["mAP"]), abs(E.stats[1] - s["AP50"]))
            worst, n = max(worst, d), n + 1
            if d > 1e-3:
                print(f"  MISMATCH {col}={g}: coco mAP {E.stats[0]:.4f} AP50 {E.stats[1]:.4f} "
                      f"| ours {s['mAP']:.4f} {s['AP50']:.4f}")
            tp, _, fn = M.counts_at(M.pool([evals[i] for i in ids]), conf)
            tp_sum, fn_sum = tp_sum + tp, fn_sum + fn
        if (tp_sum, fn_sum) != (tot[0], tot[2]):
            bad_sums.append(col)
    print(f"{n} groups checked: max |pycocotools - ours| on mAP/AP50 = {worst:.2e}")
    print("group TP/FN sums equal overall for every aspect" if not bad_sums else f"sum mismatch: {bad_sums}")


if __name__ == "__main__":
    main()
