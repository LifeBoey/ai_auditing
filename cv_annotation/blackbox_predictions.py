"""
Black box: rule-based comparison of annotations with one model's predictions
(flag_annotations.py: wrong_class, bad_location, unsupported, missing, ...).

Runs flag_annotations.py (unless --reuse finds its output) and writes the black-box contract
(see blackbox_common.py) to --out-dir:

    s_score         1 - prod(1 - severity) over the image's flags (grows with the number of issues)
    s_max_severity  the worst single flag, low_confidence excluded (doesn't grow with object count)
    flagged         flag_annotations' own verdict
    reason          "<main issue>: <why>", e.g. "wrong_class: annotated pizza; model says person 1.00 ..."

Usage:
    python blackbox_predictions.py                                   # predictions_rcnn.json
    python blackbox_predictions.py --predictions predictions_yolo.json
    python blackbox_predictions.py --swap-margin 0.3 --calibrate     # unknown options go to flag_annotations.py
"""

import argparse
import csv
import os
import subprocess
import sys

from blackbox_common import SCORES_ROOT, image_paths, predictions_tag, write_blackbox


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", default="labels.json")
    parser.add_argument("--predictions", default="predictions_rcnn.json")
    parser.add_argument("--images-dir", default="data")
    parser.add_argument("--work-dir", default=None, help="flag_annotations.py output (default: flags_<tag>)")
    parser.add_argument("--out-dir", default=None, help=f"black-box output (default: {SCORES_ROOT}/flags_<tag>)")
    parser.add_argument("--reuse", action="store_true", help="don't rerun if --work-dir already has results")
    args, passthrough = parser.parse_known_args()

    tag = predictions_tag(args.predictions)
    work_dir = args.work_dir or f"flags_{tag}"
    out_dir = args.out_dir or os.path.join(SCORES_ROOT, f"flags_{tag}")
    result = os.path.join(work_dir, "all_images.csv")

    if not (args.reuse and os.path.isfile(result)):
        cmd = [sys.executable, "-W", "ignore", "flag_annotations.py", "--labels", args.labels,
               "--predictions", args.predictions, "--images-dir", args.images_dir,
               "--out-dir", work_dir, "--num-viz", "0", *passthrough]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    with open(result, encoding="utf-8") as f:
        by_name = {r["file_name"]: r for r in csv.DictReader(f)}
    rows = []
    for name, path in image_paths(args.labels, args.images_dir).items():
        r = by_name.get(name)
        rows.append({
            "file_name": name, "file_path": path,
            "s_score": r["score"] if r else "",
            "s_max_severity": r["max_severity"] if r else "",
            "flagged": r is not None and r["flagged"] == "True",
            "reason": f"{r['main_issue']}: {r['main_reason']}" if r and r["main_issue"] else "",
        })

    write_blackbox(out_dir, rows, name=f"flags_{tag}", method="flag_annotations", group=tag,
                   signals={"s_score": "combined severity of all flags (1 - prod(1 - severity))",
                            "s_max_severity": "severity of the worst single flag (low_confidence excluded)"},
                   weights={"s_score": 0.5, "s_max_severity": 1},
                   params={"predictions": args.predictions, "labels": args.labels,
                           "flag_annotations_args": passthrough, "work_dir": work_dir})


if __name__ == "__main__":
    main()
