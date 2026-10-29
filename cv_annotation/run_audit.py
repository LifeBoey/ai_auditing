"""
Run every black box on every predictions file, then the ensemble, into one folder:

    <root>/work/      each black box's own working files (work/<black box>/)
    <root>/scores/    the black-box contract, one folder per black box run
    <root>/ensemble/  review_list.csv, all_ranked.csv (+ ensemble_nogroups/)

The black boxes aren't listed here: every blackboxes/<method>/blackbox_config.json is one (--list-methods
shows them; blackboxes/README.md says how to add one). Its "predictions" decides how it runs:
    per_model   once per --predictions file, into scores/<prefix>_<tag> (tag: predictions_rcnn.json -> rcnn)
    none        once, into scores/<prefix>: it uses only the labels and images
The ensemble gets exactly the black boxes of this run (nothing else that's in scores/), weighted
by each blackbox_config.json's "weight" unless --weights says otherwise.

By default nothing is assumed about the data: file names mean nothing. --evaluate-planted is for
test sets with planted *_wrong / *_missing images: every step that supports it marks them and
reports how many it catches, and the ensemble writes evaluation.csv.

Usage:
    python run_audit.py --labels my_labels.json --images-dir my_images    # FiftyOne dataset: audit_my_labels
    python run_audit.py --labels labels_sub500_v2.json --evaluate-planted    # test set: how much is caught
    python run_audit.py --labels labels_sub500_v2.json --reuse               # only rerun what's missing
    python run_audit.py --predictions predictions_rcnn.json --methods predictions,fiftyone,geometry
    python run_audit.py --box-args fiftyone="--min-score 0.05" --box-args geometry="--flag-z 5"
    python run_audit.py --list-methods
"""

import argparse
import json
import os
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BOXES_DIR = os.path.join(HERE, "blackboxes")
KINDS = ("per_model", "none")
ACCEPTS = ("work_dir", "evaluate_planted", "audit_name")  # optional standard arguments a blackbox_config.json can list


def find_configs():
    """{method: blackbox_config.json with defaults filled in and "script" absolute}, from blackboxes/*/blackbox_config.json."""
    boxes = {}
    for method in sorted(os.listdir(BOXES_DIR)) if os.path.isdir(BOXES_DIR) else []:
        path = os.path.join(BOXES_DIR, method, "blackbox_config.json")
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            box = {"prefix": method, "accepts": [], "weight": 1, "to_run": True, **json.load(f)}
        box["script"] = os.path.normpath(os.path.join(BOXES_DIR, method, box.get("script", "")))
        problems = []
        if not os.path.isfile(box["script"]):
            problems.append(f"script {box['script']} not found")
        if box.get("predictions") not in KINDS:
            problems.append(f'"predictions" must be one of {", ".join(KINDS)}')
        unknown = sorted(set(box["accepts"]) - set(ACCEPTS))
        if unknown:
            problems.append(f'unknown "accepts" {", ".join(unknown)} (known: {", ".join(ACCEPTS)})')
        if problems:
            raise SystemExit(f"{path}: " + "; ".join(problems))
        boxes[method] = box
    if not boxes:
        raise SystemExit(f"No black boxes: {BOXES_DIR} has no */blackbox_config.json")
    return boxes


def parse_box_args(items, boxes):
    """["fiftyone=--min-score 0.05", ...] -> {method: [arguments]}."""
    out = {}
    for item in items:
        method, sep, rest = item.partition("=")
        if not sep or method not in boxes:
            raise SystemExit(f"--box-args '{item}': expected <method>=\"<arguments>\", method one of {', '.join(boxes)}")
        out.setdefault(method, []).extend(shlex.split(rest))
    return out


def run(cmd):
    print("\n>>", " ".join(cmd), flush=True)
    path = os.environ.get("PYTHONPATH")
    env = {**os.environ, "PYTHONPATH": HERE + (os.pathsep + path if path else "")}  # blackbox_common from anywhere
    subprocess.run([sys.executable, "-W", "ignore", *cmd], check=True, env=env)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", default="labels.json")
    parser.add_argument("--images-dir", default="data")
    parser.add_argument("--predictions", default="predictions_rcnn.json,predictions_yolo.json",
                        help="comma-separated predictions files; each is its own group in the ensemble "
                             "(only per_model black boxes use them)")
    parser.add_argument("--methods", default=None,
                        help="comma-separated, see --list-methods (default: every blackbox_config.json without \"to_run\": false)")
    parser.add_argument("--list-methods", action="store_true", help="list the black boxes in blackboxes/ and exit")
    parser.add_argument("--box-args", action="append", default=[], metavar='METHOD="ARGS"',
                        help="extra arguments for one black box's script, e.g. fiftyone=\"--min-score 0.05\" (repeatable)")
    parser.add_argument("--root", default=None, help="output folder (default: audit_<labels name>)")
    parser.add_argument("--reuse", action="store_true", help="skip black boxes whose scores already exist")
    parser.add_argument("--budget", type=int, default=None, help="ensemble review list size (default: 10%% of images)")
    parser.add_argument("--combine", default="rrf")
    parser.add_argument("--weights", default="",
                        help="ensemble weights on top of each blackbox_config.json's \"weight\" (see ensemble_audit.py)")
    parser.add_argument("--audit-name", "--fiftyone-dataset", dest="audit_name", default=None,
                        help="name for state a black box keeps between runs, e.g. the FiftyOne dataset (created "
                             "from --labels if new; default: the --root folder name, so every audit has its own)")
    parser.add_argument("--evaluate-planted", action="store_true",
                        help="test sets only: treat *_wrong / *_missing images as planted errors, report how "
                             "many are caught and write ensemble/evaluation.csv (default: off)")
    args = parser.parse_args()

    boxes = find_configs()
    if args.list_methods:
        for method, box in boxes.items():
            note = "" if box["to_run"] else "  (only with --methods)"
            print(f"{method:<14} {box['predictions']:<10} weight {box['weight']:<5g}{box.get('description', '')}{note}")
        return
    methods = args.methods.split(",") if args.methods else [m for m, box in boxes.items() if box["to_run"]]
    unknown = [m for m in methods if m not in boxes]
    if unknown:
        raise SystemExit(f"Unknown method(s) {', '.join(unknown)}; available: {', '.join(boxes)} (see --list-methods)")
    box_args = parse_box_args(args.box_args, boxes)
    predictions = [p for p in args.predictions.split(",") if p]
    if not predictions and any(boxes[m]["predictions"] == "per_model" for m in methods):
        raise SystemExit("--predictions is empty, but per_model black boxes were asked for")

    from blackbox_common import check_blackbox, image_paths, predictions_tag

    root = args.root or f"audit_{os.path.splitext(os.path.basename(args.labels))[0]}"
    audit_name = args.audit_name or os.path.basename(os.path.normpath(root))
    paths = image_paths(args.labels, args.images_dir)
    budget = args.budget or max(10, len(paths) // 10)
    common = ["--labels", args.labels, "--images-dir", args.images_dir]

    jobs = [(m, p) for p in predictions for m in methods if boxes[m]["predictions"] == "per_model"]
    jobs += [(m, None) for m in methods if boxes[m]["predictions"] == "none"]
    dirs, weights = [], {}
    for method, pred in jobs:
        box = boxes[method]
        name = f"{box['prefix']}_{predictions_tag(pred)}" if pred else box["prefix"]
        out = os.path.join(root, "scores", name)
        if args.reuse and os.path.isfile(os.path.join(out, "meta.json")):
            print(f"\n-- {out} exists, skipped")
        else:
            optional = {"work_dir": ["--work-dir", os.path.join(root, "work", name)],
                        "evaluate_planted": ["--evaluate-planted"] if args.evaluate_planted else [],
                        "audit_name": ["--audit-name", audit_name]}
            run([os.path.relpath(box["script"]), *common, *(["--predictions", pred] if pred else []),
                 "--out-dir", out, *(a for key in box["accepts"] for a in optional[key]), *box_args.get(method, [])])
        meta, errors, warnings = check_blackbox(out, paths)
        for w in warnings:
            print(f"   warning, {out}: {w}")
        if errors:
            raise SystemExit(f"{out} breaks the black-box contract (see blackbox_common.py):\n  - "
                             + "\n  - ".join(errors))
        dirs.append(out)
        if box["weight"] != 1:
            weights[meta["method"]] = box["weight"]

    weights = ",".join([f"{k}={w:g}" for k, w in weights.items()] + [args.weights])  # --weights comes last: it wins
    ens = [*common, "--budget", str(budget), "--combine", args.combine, "--weights", weights,
           *(["--evaluate-planted"] if args.evaluate_planted else [])]
    ensemble = os.path.relpath(os.path.join(HERE, "ensemble_audit.py"))
    run([ensemble, *dirs, *ens, "--out-dir", os.path.join(root, "ensemble")])
    run([ensemble, *dirs, *ens, "--no-groups", "--out-dir", os.path.join(root, "ensemble_nogroups")])


if __name__ == "__main__":
    main()
