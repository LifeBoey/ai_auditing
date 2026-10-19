"""
Run every black box on every predictions file, then the ensemble, into one folder:

    <root>/work/      each method's full output (flags_<tag>/, cleanlab_<tag>_ann/)
    <root>/scores/    the black-box contract, one folder per method x predictions file
    <root>/ensemble/  review_list.csv, all_ranked.csv (+ ensemble_nogroups/)

By default nothing is assumed about the data: file names mean nothing. --evaluate-planted is for
test sets with planted *_wrong / *_missing images: every step then marks them and reports how
many it catches, and the ensemble writes evaluation.csv.

Usage:
    python run_audit.py --labels my_labels.json --images-dir my_images    # FiftyOne dataset: audit_my_labels
    python run_audit.py --labels labels_sub500_v2.json --evaluate-planted    # test set: how much is caught
    python run_audit.py --labels labels_sub500_v2.json --reuse               # only rerun what's missing
    python run_audit.py --predictions predictions_rcnn.json --methods predictions,fiftyone
"""

import argparse
import os
import subprocess
import sys

BOXES = {
    # method: (script, name prefix, has a --work-dir)
    "predictions": ("blackbox_predictions.py", "flags", True),
    "cleanlab": ("blackbox_cleanlab.py", "cleanlab", True),
    "fiftyone": ("blackbox_fiftyone.py", "fiftyone", False),
}


def run(cmd):
    print("\n>>", " ".join(cmd), flush=True)
    subprocess.run([sys.executable, "-W", "ignore", *cmd], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", default="labels.json")
    parser.add_argument("--images-dir", default="data")
    parser.add_argument("--predictions", default="predictions_rcnn.json,predictions_yolo.json",
                        help="comma-separated predictions files; each is its own group in the ensemble")
    parser.add_argument("--methods", default=",".join(BOXES), help=f"comma-separated, from {', '.join(BOXES)}")
    parser.add_argument("--root", default=None, help="output folder (default: audit_<labels name>)")
    parser.add_argument("--reuse", action="store_true", help="skip black boxes whose scores already exist")
    parser.add_argument("--budget", type=int, default=None, help="ensemble review list size (default: 10%% of images)")
    parser.add_argument("--combine", default="rrf")
    parser.add_argument("--weights", default="cleanlab=0.5", help="ensemble weights (see ensemble_audit.py)")
    parser.add_argument("--fiftyone-dataset", default=None,
                        help="FiftyOne dataset the fiftyone black box uses (created from --labels if new; "
                             "default: the --root folder name, so every audit folder has its own)")
    parser.add_argument("--evaluate-planted", action="store_true",
                        help="test sets only: treat *_wrong / *_missing images as planted errors, report how "
                             "many are caught and write ensemble/evaluation.csv (default: off)")
    args = parser.parse_args()

    from blackbox_common import image_paths, predictions_tag

    root = args.root or f"audit_{os.path.splitext(os.path.basename(args.labels))[0]}"
    fiftyone_dataset = args.fiftyone_dataset or os.path.basename(os.path.normpath(root))
    budget = args.budget or max(10, len(image_paths(args.labels, args.images_dir)) // 10)
    common = ["--labels", args.labels, "--images-dir", args.images_dir]

    for pred in args.predictions.split(","):
        tag = predictions_tag(pred)
        for method in args.methods.split(","):
            script, prefix, has_work = BOXES[method]
            out = os.path.join(root, "scores", f"{prefix}_{tag}")
            if args.reuse and os.path.isfile(os.path.join(out, "meta.json")):
                print(f"\n-- {out} exists, skipped")
                continue
            work = ["--work-dir", os.path.join(root, "work", os.path.basename(out) + ("_ann" if method == "cleanlab" else ""))]
            if method == "fiftyone":
                extra = ["--dataset", fiftyone_dataset]
            else:  # forwarded to flag_annotations.py / cleanlab_data_audit.py
                extra = ["--evaluate-planted"] if args.evaluate_planted else []
            run([script, *common, "--predictions", pred, "--out-dir", out, *(work if has_work else []), *extra])

    scores = os.path.join(root, "scores")
    dirs = sorted(os.path.join(scores, d) for d in os.listdir(scores) if os.path.isfile(os.path.join(scores, d, "meta.json")))
    ens = [*common, "--budget", str(budget), "--combine", args.combine, "--weights", args.weights,
           *(["--evaluate-planted"] if args.evaluate_planted else [])]
    run(["ensemble_audit.py", *dirs, *ens, "--out-dir", os.path.join(root, "ensemble")])
    run(["ensemble_audit.py", *dirs, *ens, "--no-groups", "--out-dir", os.path.join(root, "ensemble_nogroups")])


if __name__ == "__main__":
    main()
