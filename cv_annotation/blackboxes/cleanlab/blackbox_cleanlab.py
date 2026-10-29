"""
Black box: cleanlab ObjectLab label-quality scores for one model's predictions.

Runs only the annotation part of cleanlab_data_audit.py (--skip-quality --skip-duplicates;
unless --reuse finds its output) and writes the black-box contract (see blackbox_common.py)
to --out-dir:

    s_swap          1 - the lowest swap score over the image's annotations (wrong class)
    s_overlooked    1 - the lowest overlooked score over its predictions (missing annotation)
    s_badloc        1 - the lowest badloc score over its annotations (bad box; noisy, weight 0.25)
    s_label_issue   1 - cleanlab's label_quality_score (weight 0: see below)
    flagged         any swap / overlooked score < --flag-score, or badloc < --flag-badloc
    reason          its worst box issue, e.g. "swap: annotated pizza looks like person (0.99)"

Why per-box minimums and not label_quality_score: ObjectLab's image score blends the three
box scores, and a box whose class the model never predicted gets no swap / badloc score
(NaN, treated as clean). A planted wrong class is exactly such a box, so for a model that
predicts one class per box (YOLO) the image score ranks planted errors as the CLEANEST
images (AUC 0.11 on the 1000-image subset), while the per-box min swap score gets 0.90.

Usage:
    python blackbox_cleanlab.py                                      # predictions_rcnn.json
    python blackbox_cleanlab.py --predictions predictions_yolo.json
    python blackbox_cleanlab.py --swap-prob-threshold 0.95           # unknown options go to cleanlab_data_audit.py
"""

import argparse
import csv
import os
import subprocess
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))  # blackbox_common, when run directly
from blackbox_common import SCORES_ROOT, image_paths, predictions_tag, write_blackbox


def describe(issue):
    conf = f"{float(issue['pred_confidence']):.2f}" if issue["pred_confidence"] else "?"
    if issue["issue_type"] == "swap":
        return f"swap: annotated {issue['given_class']} looks like {issue['suggested_class']} ({conf})"
    if issue["issue_type"] == "overlooked":
        return f"overlooked: unannotated {issue['suggested_class']} ({conf})"
    iou = f" (IoU {float(issue['iou_with_match']):.2f} with the model's box)" if issue["iou_with_match"] else ""
    return f"badloc: {issue['given_class']} box badly placed{iou}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", default="labels.json")
    parser.add_argument("--predictions", default="predictions_rcnn.json")
    parser.add_argument("--images-dir", default="data")
    parser.add_argument("--work-dir", default=None, help="cleanlab_data_audit.py output (default: cleanlab_<tag>_ann)")
    parser.add_argument("--out-dir", default=None, help=f"black-box output (default: {SCORES_ROOT}/cleanlab_<tag>)")
    parser.add_argument("--reuse", action="store_true", help="don't rerun if --work-dir already has results")
    parser.add_argument("--flag-score", type=float, default=0.2, help="flag swap / overlooked box scores below this")
    parser.add_argument("--flag-badloc", type=float, default=0.05, help="flag badloc box scores below this")
    args, passthrough = parser.parse_known_args()

    tag = predictions_tag(args.predictions)
    work_dir = args.work_dir or f"cleanlab_{tag}_ann"
    out_dir = args.out_dir or os.path.join(SCORES_ROOT, f"cleanlab_{tag}")
    result = os.path.join(work_dir, "image_scores.csv")

    if not (args.reuse and os.path.isfile(result)):
        cmd = [sys.executable, "-W", "ignore", os.path.join(HERE, "cleanlab_data_audit.py"), "--labels", args.labels,
               "--predictions", args.predictions, "--images-dir", args.images_dir, "--out-dir", work_dir,
               "--skip-quality", "--skip-duplicates", "--num-viz", "0", *passthrough]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    with open(result, encoding="utf-8") as f:
        by_name = {r["file_name"]: r for r in csv.DictReader(f)}
    issues = defaultdict(list)
    box_issues = os.path.join(work_dir, "box_issues.csv")
    if os.path.isfile(box_issues):
        with open(box_issues, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                issues[r["file_name"]].append(r)

    # lowest per-box score of each kind, per image (NaN = cleanlab had no score = no evidence)
    lowest = defaultdict(dict)
    with open(os.path.join(work_dir, "all_box_scores.csv"), encoding="utf-8") as f:
        for b in csv.DictReader(f):
            kinds = ("swap_score", "badloc_score") if b["box_source"] == "annotation" else ("overlooked_score",)
            for k in kinds:
                if b[k] not in ("", "nan"):
                    lowest[b["file_name"]][k] = min(lowest[b["file_name"]].get(k, 1.0), float(b[k]))

    rows = []
    for name, path in image_paths(args.labels, args.images_dir).items():
        r = by_name.get(name)
        low = lowest.get(name, {})
        swap, over, bad = (low.get(k, 1.0) for k in ("swap_score", "overlooked_score", "badloc_score"))
        found = sorted(issues[name], key=lambda i: float(i["issue_score"]))  # lowest = worst
        reason = describe(found[0]) + (f" (+{len(found) - 1} more)" if len(found) > 1 else "") if found else ""
        rows.append({
            "file_name": name, "file_path": path,
            "s_swap": round(1 - swap, 4) if r else "",
            "s_overlooked": round(1 - over, 4) if r else "",
            "s_badloc": round(1 - bad, 4) if r else "",
            "s_label_issue": round(1 - float(r["label_quality_score"]), 4) if r else "",
            "flagged": r is not None and (min(swap, over) < args.flag_score or bad < args.flag_badloc),
            "reason": reason,
        })

    write_blackbox(out_dir, rows, name=f"cleanlab_{tag}", method="cleanlab", group=tag,
                   signals={"s_swap": "1 - lowest ObjectLab swap score over annotations",
                            "s_overlooked": "1 - lowest ObjectLab overlooked score over predictions",
                            "s_badloc": "1 - lowest ObjectLab badloc score over annotations",
                            "s_label_issue": "1 - ObjectLab label_quality_score"},
                   weights={"s_swap": 1, "s_overlooked": 1, "s_badloc": 0.25, "s_label_issue": 0},
                   params={"predictions": args.predictions, "labels": args.labels, "flag_score": args.flag_score,
                           "flag_badloc": args.flag_badloc, "cleanlab_args": passthrough, "work_dir": work_dir})


if __name__ == "__main__":
    main()
