"""
Are an audit's artifacts any use? Scores them against planted errors and writes charts + a report.

Needs a test set with planted errors: *_missing / *_wrong copies of clean images, plus the
perturbation log that make_bccdp.py writes (which boxes were deleted / relabelled). Reads, from
--audit-dir (the output of run_audit.py + ensemble_audit.py):

  scores/*/scores.csv + meta.json          image-level signals of every black box
  ensemble/all_ranked.csv                  rule scores, grouped
  ensemble_nogroups/all_ranked.csv         rule scores, every method votes
  work/flags_*/flagged_boxes.csv           box-level flags (flag_annotations)
  work/cleanlab_*_ann/box_issues.csv       box-level issues (ObjectLab)

and writes to --out-dir (default <audit-dir>/analysis): one PNG per question, the summary_*.csv
behind each, and report.html (charts embedded, opens offline).

  1 detection   ROC AUC per method / rule and planted kind
  2 effort      planted errors found vs images reviewed
  3 paired      score(perturbed copy) - score(its clean source): reacts to the error, not the image?
  4 boxes       share of planted boxes hit by a flagged box (IoU >= --iou), and noise on clean images
  5 reasons     does the reason text name the right problem?
  6 signals     AUC of every raw s_* signal
  7 redundancy  Spearman correlation and top-budget overlap between methods
  8 leakage     results split by whether the detectors were trained on the source image

Usage:
    python analyze_audit_artifacts.py --audit-dir audit_bccdp_ground_truth \\
        --perturbation-log ../DATA/bccdp_perturbation_log.json \\
        --splits-dir ../DATA/bccd_training_runs/splits
"""

import argparse
import base64
import glob
import html
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ensemble_audit import PLANTED, RULES, combine, evaluate, load_methods, parse_weights  # noqa: E402

# reference palette (light surface)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
KIND_COLOR = {"missing": SLOT[0], "wrong": SLOT[1]}
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#104281"])
DIV = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])

# reason text -> problem category; the earliest keyword in the text decides the main category
REASON_CATS = [
    ("missing box", ["overlooked", "missing"]),
    ("wrong class", ["swap", "wrong_class", "mistakenness"]),
    ("bad location", ["badloc", "bad_location"]),
    ("spurious box", ["spurious", "unsupported", "duplicate"]),
    ("low confidence", ["low_confidence"]),
]
RIGHT_CAT = {"missing": "missing box", "wrong": "wrong class"}
RIGHT_BOX_TYPES = {"missing": {"missing", "overlooked"}, "wrong": {"wrong_class", "swap"}}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": INK2, "xtick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 9.5, "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "legend.frameon": False, "savefig.dpi": 150,
})


# ---------------------------------------------------------------- ingestion

def load_truth(names, log, train_names):
    """kind (clean / missing / wrong), clean source image and train/val membership per image."""
    per = {p["image"]: p for p in log["perturbations"]}
    rows = []
    for n in names:
        kind = next((k for k, m in PLANTED.items() if m in n), "clean")
        p = per.get(n)
        if (p is None) != (kind == "clean") or (p is not None and p["type"] != kind):
            raise SystemExit(f"{n}: file name says '{kind}' but the perturbation log disagrees")
        source = p["source"] if p else n
        rows.append({"file_name": n, "kind": kind, "source": source, "in_train": source in train_names})
    truth = pd.DataFrame(rows).set_index("file_name")
    orphans = truth[(truth.kind != "clean") & ~truth.source.isin(truth.index)]
    if len(orphans):
        raise SystemExit(f"{len(orphans)} perturbed images have no clean source in the set, e.g. {orphans.index[0]}")
    return truth


def planted_boxes(log):
    rows = []
    for p in log["perturbations"]:
        for b in p.get("removed", []) + p.get("changed", []):
            rows.append({"file_name": p["image"], "kind": p["type"], "box": tuple(map(float, b["bbox_xyxy"]))})
    return pd.DataFrame(rows)


def xywh(text):
    """'x,y,w,h' or '[x, y, w, h]' -> (x1, y1, x2, y2), None when blank."""
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", str(text)) if pd.notna(text) else []
    if len(nums) != 4:
        return None
    x, y, w, h = map(float, nums)
    return (x, y, x + w, y + h)


def box_flags(audit_dir, methods):
    """{method name: DataFrame(file_name, type, boxes)} for the black boxes with box-level artifacts."""
    out = {}
    for name, (_, _, _, meta, _) in methods.items():
        work = os.path.join(audit_dir, "work", os.path.basename(meta.get("params", {}).get("work_dir", "")))
        if meta["method"] == "flag_annotations" and os.path.isfile(os.path.join(work, "flagged_boxes.csv")):
            df = pd.read_csv(os.path.join(work, "flagged_boxes.csv"))
            boxes = [[b for b in (xywh(g), xywh(p)) if b] for g, p in zip(df.gt_bbox_xywh, df.pred_bbox_xywh)]
            out[name] = pd.DataFrame({"file_name": df.file_name, "type": df.flag, "boxes": boxes})
        elif meta["method"] == "cleanlab" and os.path.isfile(os.path.join(work, "box_issues.csv")):
            df = pd.read_csv(os.path.join(work, "box_issues.csv"))
            boxes = [[b] if b else [] for b in map(xywh, df.bbox_xywh)]
            out[name] = pd.DataFrame({"file_name": df.file_name, "type": df.issue_type, "boxes": boxes})
    return out


def iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def reason_category(text):
    text = str(text or "").lower()
    hits = [(text.find(k), cat) for cat, keys in REASON_CATS for k in keys if k in text]
    return min(hits)[1] if hits else "none"


def mentions(text, cat):
    text = str(text or "").lower()
    return any(k in text for c, keys in REASON_CATS if c == cat for k in keys)


# ---------------------------------------------------------------- helpers

def short_labels(methods):
    """'cleanlab · frcnn_mbv3': method + group without the groups' common prefix."""
    groups = [m[3]["group"] for m in methods.values()]
    prefix = os.path.commonprefix(groups) if len(groups) > 1 else ""
    prefix = prefix[:prefix.rfind("_") + 1] if "_" in prefix else ""
    family = {"flag_annotations": "flags"}
    return {n: f"{family.get(m[3]['method'], m[3]['method'])} · {m[3]['group'][len(prefix):]}"
            for n, m in methods.items()}


def save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def heatmap(ax, data, cmap, norm, fmt="{:.2f}"):
    ax.imshow(data.values, cmap=cmap, norm=norm, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(data.shape[1]), data.columns, rotation=30, ha="right")
    ax.set_yticks(range(data.shape[0]), data.index)
    for s in ax.spines.values():
        s.set_visible(False)
    for (i, j), v in np.ndenumerate(data.values):
        if np.isnan(v):
            continue
        r, g, b, _ = cmap(norm(v))
        light = 0.299 * r + 0.587 * g + 0.114 * b > 0.55
        ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8, color=INK if light else "white")


def split_auc(score, truth, in_train):
    sel = truth.in_train == in_train
    pos = sel & (truth.kind != "clean")
    neg = sel & (truth.kind == "clean")
    if pos.sum() == 0 or neg.sum() == 0:
        return np.nan
    s = pd.concat([score[pos], score[neg]])
    return roc_auc_score([1] * int(pos.sum()) + [0] * int(neg.sum()), s.fillna(s.min()))


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audit-dir", default="audit_bccdp_ground_truth")
    parser.add_argument("--perturbation-log", default="../DATA/bccdp_perturbation_log.json")
    parser.add_argument("--splits-dir", default="../DATA/bccd_training_runs/splits",
                        help="folder with the detectors' train.json (COCO); omit the leakage chart if missing")
    parser.add_argument("--weights", default="cleanlab=0.5", help="as passed to ensemble_audit.py")
    parser.add_argument("--combine", choices=RULES, default="rrf", help="as passed to ensemble_audit.py")
    parser.add_argument("--rrf-k", type=float, default=60, help="as passed to ensemble_audit.py")
    parser.add_argument("--budget", type=int, default=None, help="review budget (default: rows in review_list.csv)")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU for a flagged box to count as the planted box")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # Windows consoles can't print every symbol in the tables
    out_dir = args.out_dir or os.path.join(args.audit_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    ranked = {mode: pd.read_csv(os.path.join(args.audit_dir, d, "all_ranked.csv")).set_index("file_name")
              for mode, d in (("groups", "ensemble"), ("no groups", "ensemble_nogroups"))
              if os.path.isfile(os.path.join(args.audit_dir, d, "all_ranked.csv"))}
    if not ranked:
        raise SystemExit(f"No ensemble*/all_ranked.csv in {args.audit_dir}; run ensemble_audit.py first")
    names = sorted(next(iter(ranked.values())).index)
    review_csv = os.path.join(args.audit_dir, "ensemble", "review_list.csv")
    budget = args.budget or (sum(1 for _ in open(review_csv, encoding="utf-8")) - 1 if os.path.isfile(review_csv) else 100)

    with open(args.perturbation_log, encoding="utf-8") as f:
        log = json.load(f)
    train_file = os.path.join(args.splits_dir, "train.json")
    train_names = set()
    if os.path.isfile(train_file):
        with open(train_file, encoding="utf-8") as f:
            train_names = {im["file_name"] for im in json.load(f)["images"]}
    truth = load_truth(names, log, train_names)
    planted = pd.DataFrame({k: truth.kind == k for k in PLANTED}, index=names)
    planted["any"] = planted.any(axis=1)
    n_err = int(planted["any"].sum())

    dirs = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(args.audit_dir, "scores", "*", "meta.json")))
    weights = parse_weights(args.weights)
    methods = load_methods(dirs, names, weights)
    label = short_labels(methods)
    mscore = {label[n]: m[0] for n, m in methods.items()}
    # recomputed, not read from all_ranked.csv: its 4-decimal rounding erases vote's tie-break
    rules = {mode: combine(methods, mode == "groups", args.rrf_k, weights)[0] for mode in ("groups", "no groups")}
    rscore = {f"{r} ({mode})": df[r] for mode, df in rules.items() for r in RULES}
    final = {f"final ({mode})": df[args.combine] for mode, df in rules.items()}
    pboxes = planted_boxes(log)
    bflags = {label[n]: df for n, df in box_flags(args.audit_dir, methods).items()}

    # ---- sanity checks
    checks = []
    counts = truth.kind.value_counts().to_dict()
    checks.append(f"Images: {len(names)} ({', '.join(f'{v} {k}' for k, v in counts.items())}); "
                  f"every perturbed image has its clean source in the set.")
    logged = sum(p.get("n_removed", 0) + p.get("n_changed", 0) for p in log["perturbations"])
    checks.append(f"Planted boxes: {len(pboxes)} parsed, {logged} in the log"
                  + (" (ok)" if len(pboxes) == logged else " MISMATCH"))
    rng = np.random.default_rng(0)
    shuffled = pd.Series(rng.permutation(len(names)), index=names, dtype=float)
    checks.append(f"Shuffled-score control: AUC {evaluate(shuffled, planted, budget)['any_auc']:.3f} (expect about 0.5)")
    if os.path.isfile(review_csv):
        rv = pd.read_csv(review_csv)["file_name"]
        top = ranked["groups"].sort_values("rank").index[:budget] if "groups" in ranked else rv
        same = set(rv) == set(top)
        comp = truth.loc[rv, "kind"].value_counts().to_dict()
        checks.append(f"review_list.csv ({len(rv)} images: {', '.join(f'{v} {k}' for k, v in comp.items())}) "
                      + ("matches" if same else "does NOT match") + f" the top {budget} of all_ranked.csv")
    if train_names:
        src = truth[truth.kind != "clean"]
        checks.append(f"Detector training split: {int(src.in_train.sum())} of {len(src)} perturbed images are copies "
                      f"of training images; {int((~truth.in_train & (truth.kind == 'clean')).sum())} clean images "
                      f"were held out (val).")
    for c in checks:
        print("  " + c)

    charts = []  # (title, question, png, caption, summary DataFrame)

    # ---- 1 detection power
    rows = []
    for lbl, s in {**mscore, **rscore}.items():
        ev = evaluate(s, planted, budget)
        rows.append({"what": lbl, "type": "method" if lbl in mscore else "rule", **ev})
    det = pd.DataFrame(rows).set_index("what")
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(det) + 1.4))
    y = np.arange(len(det)) + np.where(det.type == "rule", 1, 0)
    ax.barh(y - 0.2, det.missing_auc, 0.36, color=KIND_COLOR["missing"], label="missing (boxes deleted)")
    ax.barh(y + 0.2, det.wrong_auc, 0.36, color=KIND_COLOR["wrong"], label="wrong (boxes relabelled)")
    ax.axvline(0.5, color=MUTED, lw=1)
    ax.text(0.505, y.max() + 0.9, "chance", color=MUTED, fontsize=8, va="top")
    ax.set_yticks(y, det.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("ROC AUC vs clean images")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    best = det[det.type == "method"].any_auc.idxmax()
    charts.append(("Detection power", "Does any artifact rank planted errors above clean images?",
                   save(fig, out_dir, "1_detection_auc.png"),
                   f"Best single method: {best} (AUC {det.loc[best, 'any_auc']:.2f} for any planted error). "
                   f"Methods are above the gap, ensemble rules below.", det))

    # ---- 2 review effort
    def found_curve(s):
        order = s.fillna(-1).sort_values(ascending=False, kind="stable").index
        return planted["any"].loc[order].cumsum().to_numpy()

    k = np.arange(1, len(names) + 1)
    curves = {lbl: found_curve(s) for lbl, s in {**mscore, **final}.items()}
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for lbl in mscore:
        ax.plot(k, curves[lbl], color=MUTED, lw=1, alpha=0.55)
    for i, lbl in enumerate(final):
        ax.plot(k, curves[lbl], color=SLOT[i], lw=2, label=lbl)
    ax.plot(k, k * n_err / len(names), color=AXIS, lw=1.2, label="random order")
    ax.plot([], [], color=MUTED, lw=1, label="single methods")
    ax.axvline(budget, color=INK2, lw=0.8)
    hit = curves[next(iter(final))][budget - 1]
    ax.scatter([budget], [hit], s=36, color=SLOT[0], edgecolor=SURFACE, lw=1.5, zorder=5)
    ax.text(budget + 6, 2, f"review budget {budget}:\n{hit} of {n_err} found by {next(iter(final))}",
            color=INK, fontsize=8.5, va="bottom")
    ax.set_xlim(0, len(names))
    ax.set_ylim(0, n_err + 1)
    ax.set_xlabel("images reviewed, in ranking order")
    ax.set_ylabel("planted errors found")
    ax.legend(loc="lower right")
    effort = pd.DataFrame(curves, index=pd.Index(k, name="images_reviewed"))
    exp = budget * n_err / len(names)
    to_80 = int(np.argmax(curves[next(iter(final))] >= 0.8 * n_err)) + 1
    charts.append(("Review effort", "How much of the review list is worth a human's time?",
                   save(fig, out_dir, "2_review_effort.png"),
                   f"The final ranking puts {hit} of {n_err} planted errors in the top {budget} "
                   f"(random order: {exp:.1f}); finding 80% of them takes {to_80} images.",
                   effort.iloc[[budget - 1, 49, 99, 199, len(names) - 1]] if len(names) >= 200 else effort))

    # ---- 3 paired source test
    pert = truth[truth.kind != "clean"]
    paired_scores = {**mscore, **{lbl: s.rank(pct=True) for lbl, s in final.items()}}  # all on the percentile scale
    deltas = pd.DataFrame({lbl: s[pert.index].to_numpy() - s[pert.source].to_numpy()
                           for lbl, s in paired_scores.items()}, index=pert.index)
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(deltas.columns) + 1.4))
    jitter = np.random.default_rng(1).uniform(-0.16, 0.16, len(pert))
    for i, lbl in enumerate(deltas.columns):
        for kind, color in KIND_COLOR.items():
            m = (pert.kind == kind).to_numpy()
            ax.scatter(deltas[lbl][m], i + jitter[m], s=26, color=color, edgecolor=SURFACE, lw=0.8,
                       label=kind if i == 0 else None, zorder=3)
        ax.plot([deltas[lbl].median()] * 2, [i - 0.3, i + 0.3], color=INK, lw=2, zorder=4,
                label="median" if i == 0 else None)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(deltas.columns)), deltas.columns)
    ax.invert_yaxis()
    ax.set_xlim(-1, 1)
    ax.set_xlabel("score(perturbed copy) − score(clean source)   (percentile scores; > 0 = reacts to the error)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    paired = pd.DataFrame({"median_delta": deltas.median(), "share_positive": (deltas > 0).mean(),
                           **{f"median_delta_{k}": deltas[pert.kind == k].median() for k in KIND_COLOR}})
    charts.append(("Paired-source test", "Does a method react to the planted error, or just to the image?",
                   save(fig, out_dir, "3_paired_delta.png"),
                   "Each dot is one perturbed copy compared with its own clean original. Image difficulty "
                   "cancels out, so only the planted change moves the dot right.", paired))

    # ---- 4 box-level localisation
    box_rows, noise_rows = [], []
    for lbl, flags in bflags.items():
        by_image = {n: g for n, g in flags.groupby("file_name")}
        for kind in KIND_COLOR:
            pk = pboxes[pboxes.kind == kind]
            right = any_ = 0
            for _, pb in pk.iterrows():
                g = by_image.get(pb.file_name)
                if g is None:
                    continue
                hits = {t for t, bs in zip(g.type, g.boxes) if any(iou(pb.box, b) >= args.iou for b in bs)}
                any_ += bool(hits)
                right += bool(hits & RIGHT_BOX_TYPES[kind])
            # precision: right-type flags on this kind's images that sit on a planted box
            on_kind = flags[flags.file_name.isin(pk.file_name) & flags.type.isin(RIGHT_BOX_TYPES[kind])]
            pk_by = {n: list(g.box) for n, g in pk.groupby("file_name")}
            prec = np.mean([any(iou(p, b) >= args.iou for p in pk_by[f] for b in bs)
                            for f, bs in zip(on_kind.file_name, on_kind.boxes)]) if len(on_kind) else np.nan
            box_rows.append({"method": lbl, "kind": kind, "planted_boxes": len(pk), "hit_right_type": right / len(pk),
                             "hit_any_type": any_ / len(pk), "precision_right_type": prec})
        clean = flags[flags.file_name.isin(truth.index[truth.kind == "clean"])]
        noise_rows.append({"method": lbl, "flagged_boxes_per_clean_image": len(clean) / (truth.kind == "clean").sum(),
                           **{f"clean_{t}": c / (truth.kind == "clean").sum() for t, c in clean.type.value_counts().items()}})
    boxes = pd.DataFrame(box_rows)
    noise = pd.DataFrame(noise_rows).set_index("method").fillna(0)
    if len(boxes):
        bl = list(bflags)
        fig, axes = plt.subplots(1, 3, figsize=(12, 0.42 * len(bl) + 1.6), sharey=True)
        yy = np.arange(len(bl))
        for ax, kind in zip(axes, KIND_COLOR):
            b = boxes[boxes.kind == kind].set_index("method").loc[bl]
            ax.barh(yy - 0.19, b.hit_right_type, 0.36, color=KIND_COLOR[kind], label="flagged as the right problem")
            ax.barh(yy + 0.19, b.hit_any_type, 0.36, color=KIND_COLOR[kind], alpha=0.35, label="flagged as anything")
            ax.set_xlim(0, 1)
            ax.set_title(f"{kind}: {len(pboxes[pboxes.kind == kind])} planted boxes")
            ax.set_xlabel(f"share of planted boxes (IoU ≥ {args.iou:g})")
            ax.grid(axis="y", visible=False)
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42 - 0.02 * len(bl)), ncol=1, fontsize=8)
        axes[0].set_yticks(yy, bl)
        axes[0].invert_yaxis()
        axes[2].barh(yy, noise.loc[bl, "flagged_boxes_per_clean_image"], 0.5, color=SLOT[0])
        per_img = len(pboxes) / max(1, pboxes.file_name.nunique())
        axes[2].set_title(f"noise: flagged boxes per clean image\n(line: {per_img:.1f} planted boxes per perturbed image)")
        axes[2].set_xlabel("boxes flagged (all flag types)")
        axes[2].grid(axis="y", visible=False)
        axes[2].axvline(per_img, color=INK2, lw=0.8)
        axes[2].set_xlim(0, max(per_img, noise.flagged_boxes_per_clean_image.max()) * 1.08)
        fig.tight_layout()
        rm = boxes[boxes.kind == "missing"].set_index("method").hit_right_type
        rw = boxes[boxes.kind == "wrong"].set_index("method").hit_right_type
        charts.append(("Box-level localisation", "When a box is flagged, is it the right box?",
                       save(fig, out_dir, "4_box_localisation.png"),
                       f"Best at deleted boxes: {rm.idxmax()} ({rm.max():.0%}); best at relabelled boxes: "
                       f"{rw.idxmax()} ({rw.max():.0%}). FiftyOne writes no box-level artifacts, so it is not shown. "
                       "Clean BCCD labels are themselves incomplete, so some 'noise' is real errors.",
                       boxes.merge(noise[["flagged_boxes_per_clean_image"]], left_on="method", right_index=True)))

    # ---- 5 reason text
    cats = [c for c, _ in REASON_CATS] + ["none"]
    cat_color = dict(zip(cats, SLOT[:5] + [GRID]))
    reasons = {label[n]: m[2] for n, m in methods.items()}
    rrows = []
    for lbl, rs in reasons.items():
        for kind in KIND_COLOR:
            idx = truth.index[truth.kind == kind]
            main = rs[idx].map(reason_category).value_counts(normalize=True)
            rrows.append({"method": lbl, "kind": kind, **{c: main.get(c, 0.0) for c in cats},
                          "mentions_right_problem": rs[idx].map(lambda t: mentions(t, RIGHT_CAT[kind])).mean()})
    rdf = pd.DataFrame(rrows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.4 * len(reasons) + 1.8), sharey=True)
    yy = np.arange(len(reasons))
    for ax, kind in zip(axes, KIND_COLOR):
        d = rdf[rdf.kind == kind].set_index("method").loc[list(reasons)]
        left = np.zeros(len(d))
        for c in cats:
            ax.barh(yy, d[c], 0.6, left=left, color=cat_color[c], edgecolor=SURFACE, lw=1.5, label=c)
            left += d[c].to_numpy()
        ax.scatter(d.mentions_right_problem, yy, marker="|", s=180, color=INK, lw=2, zorder=4,
                   label="reason mentions the right problem somewhere")
        ax.set_xlim(0, 1)
        ax.set_title(f"{kind} images (right problem: {RIGHT_CAT[kind]})")
        ax.set_xlabel("share of the 25 images, by what the reason says first")
        ax.grid(False)
    axes[0].set_yticks(yy, list(reasons))
    axes[0].invert_yaxis()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4)
    fig.tight_layout()
    rr = rdf.groupby("method", sort=False).mentions_right_problem.mean()
    charts.append(("Reason text", "Do the reasons in the slideshow table point the reviewer at the right problem?",
                   save(fig, out_dir, "5_reason_text.png"),
                   f"Most helpful reasons: {rr.idxmax()} names the right problem for {rr.max():.0%} of perturbed images; "
                   f"least helpful: {rr.idxmin()} ({rr.min():.0%}). 'none' means the method gave no reason.", rdf))

    # ---- 6 raw signals
    srows = []
    for d in dirs:
        meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
        name = meta["name"]
        if name not in methods:
            continue
        df = pd.read_csv(os.path.join(d, "scores.csv")).set_index("file_name").reindex(names)
        for s in [c for c in df.columns if c.startswith("s_")]:
            v = pd.to_numeric(df[s], errors="coerce")
            if v.notna().sum() < 2 or v.nunique() < 2:
                ev = {"missing_auc": np.nan, "wrong_auc": np.nan, "any_auc": np.nan}
            else:
                ev = evaluate(v, planted, budget)
            w = meta.get("weights", {}).get(s, 1.0)
            srows.append({"signal": f"{label[name]} · {s[2:]} (w {w:g})", "missing": ev["missing_auc"],
                          "wrong": ev["wrong_auc"], "any": ev["any_auc"]})
    sig = pd.DataFrame(srows).set_index("signal")
    dev = np.nanmax(np.abs(sig.values - 0.5)) or 0.1
    fig, ax = plt.subplots(figsize=(6.5, 0.28 * len(sig) + 1.2))
    heatmap(ax, sig, DIV, Normalize(0.5 - dev, 0.5 + dev))
    ax.set_title("ROC AUC per raw signal (blue = informative, red = points the wrong way)")
    charts.append(("Signal usefulness", "Which signals carry information, and do the weights agree?",
                   save(fig, out_dir, "6_signal_auc.png"),
                   "w is the weight meta.json gives the signal inside its method. A signal near 0.5 with a high "
                   "weight dilutes its method; a strong signal with weight 0 is information being thrown away.", sig))

    # ---- 7 redundancy
    ms = pd.DataFrame(mscore)
    corr = ms.corr(method="spearman")
    tops = {lbl: set(s.fillna(-1).sort_values(ascending=False, kind="stable").index[:budget]) for lbl, s in mscore.items()}
    jac = pd.DataFrame([[len(tops[a] & tops[b]) / len(tops[a] | tops[b]) for b in tops] for a in tops],
                       index=list(tops), columns=list(tops))
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.5 * len(ms.columns) + 1.8))
    heatmap(axes[0], corr, SEQ, Normalize(0, 1))
    axes[0].set_title("Spearman correlation of method scores")
    heatmap(axes[1], jac, SEQ, Normalize(0, 1))
    axes[1].set_title(f"overlap (Jaccard) of each method's top {budget}")
    axes[1].set_yticklabels([])
    fig.tight_layout()
    fam = {lbl: lbl.split(" · ")[0] for lbl in mscore}
    same = [corr.loc[a, b] for a in corr for b in corr if a < b and fam[a] == fam[b]]
    diff = [corr.loc[a, b] for a in corr for b in corr if a < b and fam[a] != fam[b]]
    charts.append(("Redundancy", "Do we need all of these black boxes?",
                   save(fig, out_dir, "7_redundancy.png"),
                   f"Mean correlation within a method family (same method, different detector): {np.mean(same):.2f}; "
                   f"between families: {np.mean(diff):.2f}.", pd.concat({"spearman": corr, "jaccard_top": jac})))

    # ---- 8 leakage
    clean_val = (truth.kind == "clean") & ~truth.in_train
    if train_names and clean_val.any():
        nt, nv = int(pert.in_train.sum()), int((~pert.in_train).sum())
        nct, ncv = int(((truth.kind == "clean") & truth.in_train).sum()), int(clean_val.sum())
        lrows = []
        for lbl, s in mscore.items():  # percentile scores only; rule scores are on other scales
            lrows.append({"what": lbl,
                          "mean_score_clean_train": s[(truth.kind == "clean") & truth.in_train].mean(),
                          "mean_score_clean_val": s[clean_val].mean(),
                          "auc_train_sources": split_auc(s, truth, True),
                          "auc_val_sources": split_auc(s, truth, False)})
        leak = pd.DataFrame(lrows).set_index("what")
        panels = [(("mean_score_clean_train", "mean_score_clean_val"), "clean images: mean suspicion score",
                   "mean percentile score (lower = model agrees with the labels)",
                   (f"in the detectors' training set ({nct})", f"held out / val ({ncv})"))]
        if nv:  # perturbed copies of held-out images exist: compare detection power too
            panels.insert(0, (("auc_train_sources", "auc_val_sources"), "AUC, perturbed vs clean, within each split",
                              "ROC AUC", (f"source trained on ({nt} perturbed)", f"source held out ({nv} perturbed)")))
        fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 0.42 * len(leak) + 1.6),
                                 sharey=True, squeeze=False)
        yy = np.arange(len(leak))
        for ax, (cols, title, xl, names_) in zip(axes[0], panels):
            ax.barh(yy - 0.19, leak[cols[0]], 0.36, color=SLOT[0], label=names_[0])
            ax.barh(yy + 0.19, leak[cols[1]], 0.36, color=SLOT[1], label=names_[1])
            ax.set_xlim(0, 1)
            ax.set_title(title)
            ax.set_xlabel(xl)
            ax.grid(axis="y", visible=False)
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=1)
        axes[0][0].set_yticks(yy, leak.index)
        axes[0][0].invert_yaxis()
        fig.tight_layout()
        lower = int((leak.mean_score_clean_train < leak.mean_score_clean_val).sum())
        caption = (f"On {lower} of {len(leak)} rows, clean training images look less suspicious than clean held-out "
                   "images: the detectors agree more with labels they were trained on. ")
        if nv == 0:
            caption += (f"All {nt} perturbed images are copies of training images, so every number in this report "
                        "measures the audit on images the detectors memorised. Expect weaker results on unseen data; "
                        "plant errors in held-out images to measure that.")
        else:
            gap = (leak.auc_train_sources - leak.auc_val_sources).mean()
            caption += f"Mean AUC gap (train − val sources): {gap:+.2f}; only {nv} perturbed images are held out."
        charts.append(("Train/val leakage", "Are the results inflated because the detectors saw these images in training?",
                       save(fig, out_dir, "8_leakage.png"), caption, leak))

    # ---- verdict
    vrows = []
    for lbl in [*mscore, *final]:
        if lbl in det.index:
            ev = det.loc[lbl]
        else:
            ev = pd.Series(evaluate(final[lbl], planted, budget))
        b = boxes[boxes.method == lbl] if len(boxes) else boxes
        delta = deltas[lbl].median() if lbl in deltas else np.nan
        auc = ev["any_auc"]
        verdict = ("drop" if auc < 0.6 or (not np.isnan(delta) and delta <= 0)
                   else "keep" if auc >= 0.75 and (np.isnan(delta) or delta > 0) else "down-weight")
        vrows.append({"what": lbl, "AUC missing": ev["missing_auc"], "AUC wrong": ev["wrong_auc"],
                      f"found in top {budget}": f"{int(ev[f'any_top{budget}'])} / {n_err}",
                      "box hit (right type)": b.hit_right_type.mean() if len(b) else np.nan,
                      "median paired Δ": delta,
                      "reason names problem": rr.get(lbl, np.nan),
                      "suggestion": verdict if lbl in mscore else "—"})
    verdict = pd.DataFrame(vrows).set_index("what")

    # ---- write
    det.to_csv(os.path.join(out_dir, "summary_detection.csv"))
    effort.to_csv(os.path.join(out_dir, "summary_effort.csv"))
    paired.to_csv(os.path.join(out_dir, "summary_paired.csv"))
    deltas.to_csv(os.path.join(out_dir, "summary_paired_per_image.csv"))
    if len(boxes):
        boxes.to_csv(os.path.join(out_dir, "summary_boxes.csv"), index=False)
        noise.to_csv(os.path.join(out_dir, "summary_box_noise.csv"))
    rdf.to_csv(os.path.join(out_dir, "summary_reasons.csv"), index=False)
    sig.to_csv(os.path.join(out_dir, "summary_signals.csv"))
    corr.to_csv(os.path.join(out_dir, "summary_spearman.csv"))
    jac.to_csv(os.path.join(out_dir, "summary_jaccard_top.csv"))
    if any(c[0] == "Train/val leakage" for c in charts):
        leak.to_csv(os.path.join(out_dir, "summary_leakage.csv"))
    verdict.to_csv(os.path.join(out_dir, "summary_verdict.csv"))
    report = write_report(out_dir, args.audit_dir, checks, verdict, charts, budget)
    print(f"\nWrote {len(charts)} charts, summary_*.csv and {report}")
    pd.set_option("display.width", 200)
    print("\n" + verdict.round(3).to_string())


def table_html(df):
    return df.to_html(float_format=lambda v: f"{v:.3f}", na_rep="—", border=0, classes="tbl")


def write_report(out_dir, audit_dir, checks, verdict, charts, budget):
    sections = []
    for i, (title, question, png, caption, summary) in enumerate(charts, 1):
        with open(png, "rb") as f:
            src = "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")
        sections.append(f"""<section><h2>{i}. {html.escape(title)}</h2><p class="q">{html.escape(question)}</p>
<img src="{src}" alt="{html.escape(title)} chart"><p class="cap">{html.escape(caption)}</p>
<details><summary>Table view</summary>{table_html(summary)}</details></section>""")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Audit Artifact Usefulness</title>
<style>
:root{{color-scheme:light;--surface:{SURFACE};--page:#f9f9f7;--ink:{INK};--ink2:{INK2};--muted:{MUTED};--line:{GRID}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--page);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:24px 16px 48px}}h1{{font-size:22px;margin:0 0 4px}}
h2{{font-size:17px;margin:0 0 2px}}.sub,.q,.cap{{color:var(--ink2)}}.q{{margin:0 0 12px}}.cap{{margin:10px 0}}
section,.card{{background:var(--surface);border:1px solid rgba(11,11,11,.1);border-radius:10px;padding:18px;margin:16px 0}}
img{{max-width:100%;height:auto;display:block}}ul{{margin:6px 0;padding-left:20px}}
.tbl{{border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums;width:100%}}
.tbl th,.tbl td{{border-bottom:1px solid var(--line);padding:5px 8px;text-align:right}}
.tbl th:first-child,.tbl td:first-child{{text-align:left}}.tbl thead th{{color:var(--ink2);font-weight:600}}
.wrap{{overflow-x:auto}}details summary{{cursor:pointer;color:var(--ink2);margin-top:6px}}
</style></head><body><main>
<h1>Are the audit artifacts any use?</h1>
<p class="sub">{html.escape(os.path.abspath(audit_dir))} scored against planted errors: <b>missing</b> images
had boxes deleted, <b>wrong</b> images had boxes relabelled. Each perturbed image's clean original is in the
same set. Review budget = {budget} images (the length of review_list.csv).</p>
<div class="card"><h2>Sanity checks</h2><ul>{''.join(f'<li>{html.escape(c)}</li>' for c in checks)}</ul></div>
<div class="card"><h2>Verdict per method</h2>
<p class="q">Suggestion: <b>keep</b> if AUC (any error) ≥ 0.75 and the paired Δ is positive; <b>drop</b> if AUC &lt; 0.6
or the paired Δ ≤ 0; otherwise <b>down-weight</b>. These thresholds are rough rules of thumb, not tuned.</p>
<div class="wrap">{table_html(verdict)}</div></div>
{''.join(sections)}
</main></body></html>"""
    path = os.path.join(out_dir, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return path


if __name__ == "__main__":
    main()
