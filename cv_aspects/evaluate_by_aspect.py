"""
evaluate_by_aspect.py -- stage 2: how does detector performance depend on each aspect?

For every aspect in tags.csv (columns <aspect>__group) and every group of it, computes on that
group's images: mAP@[.5:.95], AP50 (+ bootstrap 95% CI), per-class AP50, and at one fixed score
threshold (same for every group) TP/FP/FN, precision, recall, F1 with Wilson 95% CIs. Each group
is compared with the rest of the data (two-proportion z-test on recall and precision,
Benjamini-Hochberg over all groups). Aspects are ranked by how much AP50 varies across groups.

    python evaluate_by_aspect.py --preds ../DATA/prediction_jsons/predictions_bccd_frcnn_mbv3.json \
        --labels ../DATA/bccd_ground_truth.json --tags aspects_out/tags.csv

Outputs (in --out, default: next to tags.csv, sub-folder named after the predictions file):
    metrics_by_aspect.csv / .json   one row per (aspect, group), plus an "overall" row
    aspect_ranking.csv              aspects sorted by AP50 spread across groups
    plots/<aspect>.png, plots/ranking.png
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

import metrics as M
from io_utils import load_gt, load_preds

GROUP_SUFFIX = "__group"


def group_order(tags_meta, aspect, values):
    order = (tags_meta.get("aspects", {}).get(aspect, {}).get("groups") or [])
    return [g for g in order if g in values] + sorted(v for v in values if v not in order)


def evaluate_group(evals, idx, classes, conf, t, n_boot, rng, total):
    p = M.pool([evals[i] for i in idx])
    s = M.summary(p, classes)
    tp, fp, fn = M.counts_at(p, conf, t)
    prec, rec, f1 = M.prf(tp, fp, fn)
    row = dict(n_images=len(idx), n_gt=tp + fn, n_pred=tp + fp, TP=tp, FP=fp, FN=fn,
               mAP=s["mAP"], AP50=s["AP50"], precision=prec, recall=rec, F1=f1)
    row["recall_lo"], row["recall_hi"] = M.wilson(tp, tp + fn)
    row["precision_lo"], row["precision_hi"] = M.wilson(tp, tp + fp)
    if n_boot and len(idx) > 1:
        b = [M.ap50(M.pool([evals[i] for i in rng.choice(idx, len(idx))]), classes) for _ in range(n_boot)]
        row["AP50_lo"], row["AP50_hi"] = np.nanpercentile(b, [2.5, 97.5])
    else:
        row["AP50_lo"] = row["AP50_hi"] = np.nan
    if total is not None:                                   # compare with the rest of the data
        rtp, rfp, rfn = total[0] - tp, total[1] - fp, total[2] - fn
        row["recall_p"] = M.two_prop_p(tp, tp + fn, rtp, rtp + rfn)
        row["precision_p"] = M.two_prop_p(tp, tp + fp, rtp, rtp + rfp)
    for c, v in s["per_class_AP50"].items():
        row[f"AP50_{classes[c]}"] = v
    return row


def plot_aspect(df, aspect, overall, path, min_group, model):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BAR, BAR_LOW, INK, INK2, GRID = "#2a78d6", "#86b6ef", "#0b0b0b", "#52514e", "#e4e3df"
    d = df[df["aspect"] == aspect]
    x = np.arange(len(d))
    panels = [("AP50", "AP50_lo", "AP50_hi", "AP50 (bootstrap 95% CI)"),
              ("recall", "recall_lo", "recall_hi", "Recall (Wilson 95% CI)"),
              ("precision", "precision_lo", "precision_hi", "Precision (Wilson 95% CI)")]
    fig, axs = plt.subplots(1, 3, figsize=(max(9, 1.0 * len(d) * 3 + 2), 3.6), sharey=False)
    for ax, (m, lo, hi, title) in zip(axs, panels):
        v = d[m].to_numpy(float)
        err = np.vstack([v - d[lo].to_numpy(float), d[hi].to_numpy(float) - v])
        colors = [BAR if n >= min_group else BAR_LOW for n in d["n_images"]]
        ax.bar(x, v, width=0.6, color=colors, edgecolor="white", linewidth=2, zorder=2)
        # a percentile bootstrap CI need not contain the point estimate, so clip at 0
        ax.errorbar(x, v, yerr=np.clip(np.nan_to_num(err), 0, None), fmt="none", ecolor=INK2, elinewidth=1, capsize=3, zorder=3)
        ax.axhline(overall[m], color=INK2, linestyle="--", linewidth=1, zorder=1)
        ax.text(len(d) - 0.5, overall[m], f" all {overall[m]:.2f}", color=INK2, fontsize=7, va="bottom", ha="right")
        for xi, vi in zip(x, v):
            if np.isfinite(vi):
                ax.text(xi, 0.02, f"{vi:.2f}", color="white", fontsize=7, ha="center", va="bottom", zorder=4)
        lo_y = np.nanmin(d[lo].to_numpy(float)) if np.isfinite(d[lo]).any() else 0
        ax.set_ylim(0, 1.05 if lo_y > 0.3 else max(1.05, np.nanmax(v) * 1.1))
        ax.set_xticks(x, [f"{g}\nn={n}" for g, n in zip(d["group"], d["n_images"])], fontsize=7,
                      rotation=0 if len(d) <= 4 else 30, ha="center" if len(d) <= 4 else "right")
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=7)
    note = f"lighter bars: fewer than {min_group} images" if (d["n_images"] < min_group).any() else ""
    fig.suptitle(f"{model} - performance by {aspect}", fontsize=10, color=INK, x=0.01, ha="left")
    if note:
        fig.text(0.99, 0.01, note, fontsize=7, color=INK2, ha="right")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def plot_ranking(rank, path, model):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = rank.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(r) + 1.2))
    ax.barh(np.arange(len(r)), r["AP50_spread"], height=0.6, edgecolor="white", linewidth=2,
            color=["#2a78d6" if t else "#86b6ef" for t in r["trusted"]])
    for i, (v, w, t) in enumerate(zip(r["AP50_spread"], r["worst_group"], r["trusted"])):
        ax.text(v, i, f" {v:.3f}  (worst: {w}){'' if t else '  - small groups, not ranked'}",
                va="center", fontsize=7, color="#52514e")
    ax.set_yticks(np.arange(len(r)), r["aspect"], fontsize=8)
    ax.set_xlabel("AP50 spread across groups (max - min)", fontsize=8, color="#52514e")
    ax.set_xlim(0, max(r["AP50_spread"].max() * 1.6, 0.01))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors="#52514e", labelsize=7)
    ax.set_title(f"{model} - which aspects matter most", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--tags", required=True, help="tags.csv from split_by_aspect.py (file_name + *__group columns)")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU for precision/recall (one of 0.50, 0.55, ..., 0.95)")
    ap.add_argument("--conf", default="auto", help="score threshold for P/R/F1, or 'auto' = best F1 on all images")
    ap.add_argument("--min-group", type=int, default=20, help="groups with fewer images are flagged low_n")
    ap.add_argument("--bootstrap", type=int, default=200, help="resamples for the AP50 CI (0 = off)")
    ap.add_argument("--out", help="output folder")
    args = ap.parse_args()

    model = os.path.splitext(os.path.basename(args.preds))[0]
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.tags)), model)
    os.makedirs(os.path.join(out, "plots"), exist_ok=True)
    t = int(np.argmin(np.abs(M.IOU_THRS - args.iou)))
    if abs(M.IOU_THRS[t] - args.iou) > 1e-6:
        raise SystemExit(f"--iou must be one of {np.round(M.IOU_THRS, 2).tolist()}")

    gt, anns = load_gt(args.labels)
    preds = load_preds(args.preds, gt)
    classes = {c["id"]: c["name"] for c in gt["categories"]}
    fid = {im["file_name"]: im["id"] for im in gt["images"]}

    tags = pd.read_csv(args.tags, dtype=str, keep_default_na=False)
    meta_path = os.path.splitext(args.tags)[0] + ".json"
    tags_meta = json.load(open(meta_path, encoding="utf-8")) if os.path.isfile(meta_path) else {}
    unknown = ~tags["file_name"].isin(fid)
    if unknown.any():
        print(f"[warn] {unknown.sum()} tagged images are not in the GT; ignored")
    tags = tags[~unknown].reset_index(drop=True)
    untagged = len(fid) - len(tags)
    if untagged:
        print(f"[warn] {untagged} GT images have no tags; they are left out of every number below")

    ids = sorted(fid[f] for f in tags["file_name"])                  # COCOeval order: sorted imgIds
    pos = {iid: k for k, iid in enumerate(ids)}
    print(f"matching {len(ids)} images ...")
    evals = [M.evaluate_image(anns.get(i, []), preds.get(i, [])) for i in ids]
    row_of = np.array([pos[fid[f]] for f in tags["file_name"]])     # tags row -> evals index

    allp = M.pool(evals)
    if args.conf == "auto":
        conf, f1 = M.best_f1_conf(allp, t)
        print(f"score threshold = {conf:.4f} (best F1 {f1:.3f} on all images at IoU {M.IOU_THRS[t]:.2f})")
    else:
        conf = float(args.conf)
    rng = np.random.default_rng(0)
    overall = evaluate_group(evals, np.arange(len(ids)), classes, conf, t, args.bootstrap, rng, None)
    total = (overall["TP"], overall["FP"], overall["FN"])
    print(f"overall: mAP {overall['mAP']:.4f}  AP50 {overall['AP50']:.4f}  P {overall['precision']:.3f}  "
          f"R {overall['recall']:.3f}  F1 {overall['F1']:.3f}")

    aspects = [c[:-len(GROUP_SUFFIX)] for c in tags.columns if c.endswith(GROUP_SUFFIX)]
    rows = [dict(aspect="overall", group="all", **overall)]
    for a in aspects:
        col = tags[a + GROUP_SUFFIX]
        for g in group_order(tags_meta, a, set(col)):
            idx = row_of[(col == g).to_numpy()]
            rows.append(dict(aspect=a, group=g, **evaluate_group(evals, idx, classes, conf, t,
                                                                 args.bootstrap, rng, total)))
        print(f"  {a}: {col.nunique()} groups")
    df = pd.DataFrame(rows)
    df["low_n"] = df["n_images"] < args.min_group
    for m in ("recall", "precision"):
        df[f"{m}_q"] = M.bh(df[f"{m}_p"].to_numpy(float))
    for m in ("mAP", "AP50", "recall", "precision", "F1"):
        df[f"d_{m}"] = df[m] - overall[m]
    df.insert(2, "score_threshold", conf)
    df.insert(3, "iou", float(M.IOU_THRS[t]))

    # ranking: spread over groups that are big enough to trust
    rank = []
    for a in aspects:
        d = df[(df["aspect"] == a) & ~df["low_n"]]
        trusted = len(d) >= 2
        if not trusted:                       # spread would rest on a group below --min-group
            d = df[df["aspect"] == a]
        w = d.loc[d["AP50"].idxmin()]
        rank.append(dict(aspect=a, trusted=trusted, n_groups=len(d), AP50_spread=d["AP50"].max() - d["AP50"].min(),
                         recall_spread=d["recall"].max() - d["recall"].min(),
                         worst_group=w["group"], worst_AP50=w["AP50"], worst_d_recall=w["d_recall"],
                         worst_recall_q=w["recall_q"], best_group=d.loc[d["AP50"].idxmax(), "group"]))
    rank = pd.DataFrame(rank).sort_values(["trusted", "AP50_spread"], ascending=[False, False])

    df.round(4).to_csv(os.path.join(out, "metrics_by_aspect.csv"), index=False)
    df.round(4).to_json(os.path.join(out, "metrics_by_aspect.json"), orient="records", indent=1)
    rank.round(4).to_csv(os.path.join(out, "aspect_ranking.csv"), index=False)
    for a in aspects:
        plot_aspect(df, a, overall, os.path.join(out, "plots", f"{a}.png"), args.min_group, model)
    if len(rank):
        plot_ranking(rank, os.path.join(out, "plots", "ranking.png"), model)

    print("\naspects ranked by AP50 spread across groups:")
    print(rank[["aspect", "trusted", "n_groups", "AP50_spread", "worst_group", "worst_AP50", "worst_d_recall", "worst_recall_q"]]
          .round(3).to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
