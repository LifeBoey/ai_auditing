"""
failure_analysis.py -- where does a detector fail, how, and why?
================================================================

Framework-agnostic: reads the plain-file artifacts written by the audit script
(see ARTIFACT CONTRACT) and writes units + findings that failure_explorer.py shows.
It never imports FiftyOne, so a Detectron2 / Ultralytics / MMDetection run can feed
it the same files.

What it does
------------
1. Matching + error taxonomy at one operating point (TIDE-style, but per object):
     every ground-truth object -> detected | confused | mislocalized | missed
         missed is split into WHY: low_score_correct, low_score_wrong_class,
         low_score_loc, shadowed, no_proposal
     every prediction (score >= threshold) -> tp | dupe | cls | loc | both | bkg
2. Attributes for every unit (size, occlusion, crowding, truncation, position, the
   image-quality measurements from the audit, CLIP-density, scene cluster).
3. Slice search: every attribute value, tested for an elevated failure rate with a
   one-sided z-test, Benjamini-Hochberg FDR control, a minimum lift and size, and
   de-duplication of near-identical slices.
4. A cause battery per slice. Each hypothesis returns supports / partial / against /
   n/a with its numbers. Nothing is hidden: "against" results are shown too.
5. Overview numbers: error budget, threshold sweep, per-class table, dose-response
   curves, confidence calibration.

What it deliberately does NOT do (needs re-running the model, so it is listed as
"not run" on every finding): perturbation tests, background/shortcut masking,
Grad-CAM / D-RISE.

ARTIFACT CONTRACT (all in one directory)
----------------------------------------
images.csv      image_id, filepath, width, height, [optional numeric/bool columns:
                blurriness, brightness, contrast, ..., low_quality, is_duplicate,
                issues (";"-joined)]
gt.csv          image_id, gt_idx, label, x, y, w, h  (relative, top-left origin),
                iscrowd, suspect (0/1 = label flagged as likely wrong upstream),
                [optional suspect_reason; any other column, e.g. det_id, is carried through]
preds.csv       image_id, pred_idx, label, x, y, w, h, confidence  [+ extra columns, carried through]
embeddings.npz  image_ids, emb          (image-level embeddings, e.g. CLIP)
points2d.npz    image_ids, xy           (optional 2D projection for the failure map)
reference_emb.npz  emb                  (optional: TRAINING-set embeddings, same model)

OUTPUTS (same directory)
------------------------
gt_units.csv, pred_units.csv   every unit with its outcome, reason, attributes and explanation
findings.json                  ranked finding cards (recall + precision) with their hypotheses
finding_members.csv            finding_id, side, image_id, idx, failed: every unit in each slice
overview.json, clusters.json, failure_map.json

Run:  python failure_analysis.py --dir <artifact dir> [--conf 0.5]
Test: python check_demo.py   (recovers the patterns demo_data.py plants)
"""

import argparse
import json
import math
import os
import time
from collections import Counter

import numpy as np
import pandas as pd

CFG = dict(
    conf_thresh=0.5,   # operating point: predictions below this count as "not predicted"
    iou_fg=0.5,        # IoU needed to count as locating an object
    iou_bg=0.1,        # below this a box is "background"
    shadow_iou=0.3,    # a same-class detection overlapping a missed GT this much "shadows" it
    min_slice=30,      # smallest slice we will report
    min_fail=5,        # fewest failures a slice needs
    fdr=0.05,          # Benjamini-Hochberg level
    min_lift=1.25,     # slice failure rate / rest-of-data failure rate
    min_adj_lift=1.15, # observed / expected failures once earlier findings are accounted for
    min_share=0.01,    # a finding must explain at least this share of ALL failures (practical relevance)
    min_diff=0.05,     # ...or an absolute gap of this many points (lift is capped when the baseline is high)
    top_k=10,          # findings per side
    knn=10,            # neighbours for the embedding-density measure
    n_clusters=None,   # None -> chosen from the number of images
    seed=0,
    sweep=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
)

SIZE_EDGES = (32 ** 2, 96 ** 2)   # COCO small / medium / large, applied to box area
IMG_NUM = ["aspect_ratio", "blurriness", "brightness", "contrast", "entropy",
           "min_exposure", "max_exposure", "salt_and_pepper", "saturation",
           "vignetting", "min_side", "nn_distance", "uniqueness"]
IMG_BOOL = ["low_quality", "is_duplicate"]
GT_BASE_COLS = {"image_id", "gt_idx", "label", "x", "y", "w", "h", "iscrowd", "suspect"}
PRED_BASE_COLS = {"image_id", "pred_idx", "label", "x", "y", "w", "h", "confidence"}

ATTR_NAMES = {
    "label": "class", "size_bucket": "object size", "aspect_bucket": "box shape",
    "truncated": "touches image edge", "overlap_bucket": "overlap with other objects",
    "n_obj_bucket": "objects in image", "n_preds_bucket": "detections in image",
    "position": "position in frame", "scene_cluster": "scene group",
    "score_bin": "confidence", "area_px": "box area", "overlap_max": "overlap",
    "center_dist": "distance from center", "sparsity": "density in CLIP space",
    "low_quality": "flagged low quality", "suspect": "flagged as a suspect label", "is_duplicate": "near-duplicate image",
}
FAMILY = {"size_bucket": "size", "area_px": "size", "area_px_q": "size",
          "overlap_bucket": "occlusion", "overlap_max": "occlusion", "overlap_max_q": "occlusion",
          "n_obj_bucket": "crowding", "n_preds_bucket": "crowding",
          "position": "position", "center_dist": "position", "center_dist_q": "position"}
ORDINAL_ATTRS = {"size_bucket", "overlap_bucket", "n_obj_bucket", "n_preds_bucket", "position"}
NUM_FAMILY_COL = {"size": "area_px", "occlusion": "overlap_max", "crowding": "n_objects"}
CLUSTER_WORDS = {"brightness": ("darker", "brighter"), "blurriness": ("blurrier", "sharper"),
                 "contrast": ("lower contrast", "higher contrast"),
                 "saturation": ("less saturated", "more saturated"),
                 "entropy": ("less detailed", "more detailed"),
                 "salt_and_pepper": ("cleaner", "noisier")}
NOT_RUN = ["Perturbation sensitivity (re-run the model on blurred / noisy / darkened copies)",
           "Background and shortcut reliance (mask the object or swap the background)",
           "Attribution on example failures (Grad-CAM, D-RISE)"]


# ============================================================
# geometry
# ============================================================
def _xyxy(df, W, H):
    if len(df) == 0:
        return np.zeros((0, 4))
    x1 = df["x"].to_numpy(float) * W
    y1 = df["y"].to_numpy(float) * H
    return np.stack([x1, y1, x1 + df["w"].to_numpy(float) * W,
                     y1 + df["h"].to_numpy(float) * H], 1)


def _area(b):
    return np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)


def overlaps(a, b):
    """(IoU, intersection area) matrices, shape (len(a), len(b))."""
    if len(a) == 0 or len(b) == 0:
        z = np.zeros((len(a), len(b)))
        return z, z
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    union = _area(a)[:, None] + _area(b)[None, :] - inter
    return inter / np.maximum(union, 1e-9), inter


def _size_bucket(a):
    return "small" if a < SIZE_EDGES[0] else "medium" if a < SIZE_EDGES[1] else "large"


def _aspect_bucket(r):
    return "tall" if r < 0.5 else "wide" if r > 2 else "square-ish"


def _overlap_bucket(o):
    return "none" if o < 0.05 else "light" if o < 0.3 else "heavy"


def _count_bucket(n):
    return "1" if n <= 1 else "2-5" if n <= 5 else "6-15" if n <= 15 else "16+"


def _pos_bucket(d):
    return "center" if d < 0.3 else "middle" if d < 0.6 else "edge"


def _shape_attrs(b, W, H):
    """Geometry attributes shared by GT boxes and predictions."""
    bw = np.clip(b[:, 2] - b[:, 0], 1e-6, None)
    bh = np.clip(b[:, 3] - b[:, 1], 1e-6, None)
    area = bw * bh
    cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    cd = np.hypot(2 * (cx / W - 0.5), 2 * (cy / H - 0.5)) / math.sqrt(2)
    trunc = (b[:, 0] <= 2) | (b[:, 1] <= 2) | (b[:, 2] >= W - 2) | (b[:, 3] >= H - 2)
    return dict(area_px=area, size_bucket=[_size_bucket(a) for a in area],
                aspect_bucket=[_aspect_bucket(r) for r in bw / bh], truncated=trunc,
                center_dist=cd, position=[_pos_bucket(d) for d in cd])


# ============================================================
# matching + taxonomy for one image
# ============================================================
def analyze_image(W, H, g, p, cfg, light=False):
    """g: GT rows (label,x,y,w,h,iscrowd,suspect,gt_idx); p: predictions sorted by
    descending confidence (label,x,y,w,h,confidence,pred_idx).
    light=True returns only counts (used for the threshold sweep)."""
    thr, fg, bg = cfg["conf_thresh"], cfg["iou_fg"], cfg["iou_bg"]
    ng, npd = len(g), len(p)
    gb, pb = _xyxy(g, W, H), _xyxy(p, W, H)
    glab, plab = g["label"].to_numpy(), p["label"].to_numpy()
    psc = p["confidence"].to_numpy(float)
    crowd = g["iscrowd"].to_numpy(bool) if ng else np.zeros(0, bool)
    valid = ~crowd
    iou, inter = overlaps(pb, gb)
    same = plab[:, None] == glab[None, :] if (ng and npd) else np.zeros((npd, ng), bool)
    ops = psc >= thr

    # greedy matching, highest score first (COCO-style, same class only)
    gt_match = np.full(ng, -1)
    pm = np.full(npd, -1)
    ptype = np.array(["subthr"] * npd, dtype=object)
    for i in np.flatnonzero(ops):
        cand = np.flatnonzero(same[i] & valid & (gt_match < 0) & (iou[i] >= fg))
        if cand.size:
            j = cand[np.argmax(iou[i, cand])]
            gt_match[j], pm[i], ptype[i] = i, j, "tp"

    # classify every unmatched prediction above the threshold
    parea = _area(pb)
    p_link, p_iou = np.full(npd, -1), np.zeros(npd)
    crowd_idx = np.flatnonzero(crowd)
    for i in np.flatnonzero(ops & (pm < 0)):
        if crowd_idx.size and np.any(same[i, crowd_idx] &
                                     (inter[i, crowd_idx] / max(parea[i], 1e-9) >= 0.5)):
            ptype[i] = "ignored"
            continue
        s_idx = np.flatnonzero(same[i] & valid)
        d_idx = np.flatnonzero(~same[i] & valid)
        bs = iou[i, s_idx].max() if s_idx.size else -1.0
        bd = iou[i, d_idx].max() if d_idx.size else -1.0
        if bs >= fg:
            t, j, v = "dupe", s_idx[np.argmax(iou[i, s_idx])], bs
        elif bd >= fg:
            t, j, v = "cls", d_idx[np.argmax(iou[i, d_idx])], bd
        elif bs >= bg:
            t, j, v = "loc", s_idx[np.argmax(iou[i, s_idx])], bs
        elif bd >= bg:
            t, j, v = "both", d_idx[np.argmax(iou[i, d_idx])], bd
        else:
            t, j, v = "bkg", -1, max(bs, bd, 0.0)
        ptype[i], p_link[i], p_iou[i] = t, j, v

    if light:
        fps = Counter(t for t in ptype if t in ("dupe", "cls", "loc", "both", "bkg"))
        return dict(n_gt=int(valid.sum()), tp=int((ptype == "tp").sum()), **fps)

    # extra input columns (e.g. det_id, suspect_reason) ride along so results can be traced back
    g_extra = [c for c in g.columns if c not in GT_BASE_COLS]
    p_extra = [c for c in p.columns if c not in PRED_BASE_COLS]

    # ----- ground-truth units
    gt_rows = []
    if ng:
        sh = _shape_attrs(gb, W, H)
        gi, ginter = overlaps(gb, gb)
        np.fill_diagonal(ginter, 0)
        occ = (ginter[:, valid].max(1) / np.maximum(_area(gb), 1e-9)) if valid.any() else np.zeros(ng)
        n_obj = int(valid.sum())
    for j in range(ng):
        row = dict(image_id=g["image_id"].iat[j], gt_idx=int(g["gt_idx"].iat[j]), label=glab[j],
                   x=float(g["x"].iat[j]), y=float(g["y"].iat[j]), w=float(g["w"].iat[j]),
                   h=float(g["h"].iat[j]), crowd=bool(crowd[j]), suspect=bool(g["suspect"].iat[j]),
                   area_px=float(sh["area_px"][j]), size_bucket=sh["size_bucket"][j],
                   aspect_bucket=sh["aspect_bucket"][j], truncated=bool(sh["truncated"][j]),
                   center_dist=float(sh["center_dist"][j]), position=sh["position"][j],
                   overlap_max=float(min(occ[j], 1.0)), overlap_bucket=_overlap_bucket(occ[j]),
                   n_objects=n_obj, n_obj_bucket=_count_bucket(n_obj),
                   outcome="ignored", why="", iou=np.nan, pred_label="", pred_score=np.nan,
                   best_score_same=np.nan, best_iou_same=np.nan)
        for c in g_extra:
            row.setdefault(c, g[c].iat[j])
        if crowd[j]:
            row["why"] = "crowd region (excluded from scoring)"
            gt_rows.append(row)
            continue
        ci, cs = iou[:, j], same[:, j] if npd else np.zeros(0, bool)
        if npd:
            sm = cs & (ci >= bg)
            if sm.any():
                row["best_iou_same"] = float(ci[sm].max())
            fm = cs & (ci >= fg)
            if fm.any():
                row["best_score_same"] = float(psc[fm].max())
        i = gt_match[j]
        if i >= 0:
            row.update(outcome="detected", iou=float(ci[i]), pred_label=plab[i], pred_score=float(psc[i]))
        elif npd:
            dm = ops & ~cs & (ci >= fg)
            lm = ops & cs & (ci >= bg) & (ci < fg)
            low = ~ops
            if dm.any():
                k = np.flatnonzero(dm)[np.argmax(ci[dm])]
                row.update(outcome="confused", iou=float(ci[k]), pred_label=plab[k], pred_score=float(psc[k]))
            elif lm.any():
                k = np.flatnonzero(lm)[np.argmax(ci[lm])]
                row.update(outcome="mislocalized", iou=float(ci[k]), pred_label=plab[k], pred_score=float(psc[k]))
            else:
                row["outcome"] = "missed"
                a, b, c = low & cs & (ci >= fg), low & ~cs & (ci >= fg), low & cs & (ci >= bg)
                shadow = (ptype == "tp") & cs & (ci >= cfg["shadow_iou"]) & (pm != j)
                for why, m in (("low_score_correct", a), ("low_score_wrong_class", b),
                               ("low_score_loc", c), ("shadowed", shadow)):
                    if m.any():
                        k = np.flatnonzero(m)[np.argmax(ci[m])]
                        row.update(why=why, iou=float(ci[k]), pred_label=plab[k], pred_score=float(psc[k]))
                        break
                else:
                    row["why"] = "no_proposal"
        else:
            row.update(outcome="missed", why="no_proposal")
        gt_rows.append(row)

    # ----- prediction units (only those at or above the threshold, minus crowd-ignored)
    pred_rows = []
    if npd:
        sp = _shape_attrs(pb, W, H)
        n_ops = int((ops & (ptype != "ignored")).sum())
    for i in np.flatnonzero(ops & (ptype != "ignored")):
        t = ptype[i]
        row = dict(image_id=p["image_id"].iat[i], pred_idx=int(p["pred_idx"].iat[i]), label=plab[i],
                   x=float(p["x"].iat[i]), y=float(p["y"].iat[i]), w=float(p["w"].iat[i]),
                   h=float(p["h"].iat[i]), score=float(psc[i]), outcome=t,
                   iou=float(iou[i, pm[i]]) if t == "tp" else float(p_iou[i]),
                   gt_label=(glab[pm[i]] if t == "tp" else glab[p_link[i]] if p_link[i] >= 0 else ""),
                   area_px=float(sp["area_px"][i]), size_bucket=sp["size_bucket"][i],
                   aspect_bucket=sp["aspect_bucket"][i], truncated=bool(sp["truncated"][i]),
                   center_dist=float(sp["center_dist"][i]), position=sp["position"][i],
                   n_preds=n_ops, n_preds_bucket=_count_bucket(n_ops),
                   score_bin=f"{min(int(psc[i] * 10), 9) / 10:.1f}-{min(int(psc[i] * 10), 9) / 10 + 0.1:.1f}")
        for c in p_extra:
            row.setdefault(c, p[c].iat[i])
        pred_rows.append(row)
    # sub-threshold predictions are kept for display only
    sub_rows = []
    for i in np.flatnonzero(~ops):
        row = dict(image_id=p["image_id"].iat[i], pred_idx=int(p["pred_idx"].iat[i]),
                   label=plab[i], x=float(p["x"].iat[i]), y=float(p["y"].iat[i]),
                   w=float(p["w"].iat[i]), h=float(p["h"].iat[i]), score=float(psc[i]),
                   outcome="subthr")
        for c in p_extra:
            row.setdefault(c, p[c].iat[i])
        sub_rows.append(row)
    return gt_rows, pred_rows, sub_rows


def _ctx_words(r):
    bits = [f"{r['size_bucket']} ({r['area_px']:.0f} px²)"]
    if r["overlap_max"] >= 0.3:
        bits.append(f"{r['overlap_max']:.0%} covered by neighbours")
    if r["truncated"]:
        bits.append("cut off by the image edge")
    bits.append(f"{r['n_objects']} objects in the image")
    return ", ".join(bits)


def explain_gt(r, cfg):
    thr, fg, bg = cfg["conf_thresh"], cfg["iou_fg"], cfg["iou_bg"]
    o, w, lab = r["outcome"], r["why"], r["label"]
    if o == "detected":
        return f"Detected: matched a '{lab}' prediction (score {r['pred_score']:.2f}, IoU {r['iou']:.2f})."
    if o == "ignored":
        return "Marked as a crowd region; not scored."
    if o == "confused":
        return (f"Found but mislabeled: the model's box overlaps it (IoU {r['iou']:.2f}) but says "
                f"'{r['pred_label']}' at score {r['pred_score']:.2f} instead of '{lab}'.")
    if o == "mislocalized":
        return (f"Right class, wrong box: '{lab}' predicted at score {r['pred_score']:.2f}, but "
                f"IoU is only {r['iou']:.2f} (needs {fg}). Object: {_ctx_words(r)}.")
    why = {
        "low_score_correct": f"The model did propose a correct '{lab}' box (IoU {r['iou']:.2f}) but scored it "
                             f"{r['pred_score']:.2f}, under the {thr} threshold.",
        "low_score_wrong_class": f"The model proposed a box here (IoU {r['iou']:.2f}) but as "
                                 f"'{r['pred_label']}' with a low score of {r['pred_score']:.2f}.",
        "low_score_loc": f"Only a weak, poorly aligned '{lab}' box exists (IoU {r['iou']:.2f}, score "
                         f"{r['pred_score']:.2f}).",
        "shadowed": f"A neighbouring '{lab}' detection overlaps this object (IoU {r['iou']:.2f}); it was "
                    f"likely suppressed as a duplicate (NMS) or absorbed by the neighbour.",
        "no_proposal": f"Nothing overlaps this object (no prediction above IoU {bg}); the model never proposed it.",
    }[w]
    return f"Missed: {why} Object: {_ctx_words(r)}."


def explain_pred(r, cfg):
    o = r["outcome"]
    if o == "tp":
        return f"Correct: '{r['label']}' at {r['score']:.2f}, IoU {r['iou']:.2f} with ground truth."
    if o == "dupe":
        return (f"Duplicate: a second '{r['label']}' box (score {r['score']:.2f}) on an object that is already "
                f"detected (IoU {r['iou']:.2f}).")
    if o == "cls":
        return (f"Wrong class: box fits a ground-truth '{r['gt_label']}' (IoU {r['iou']:.2f}) but the model "
                f"says '{r['label']}' at {r['score']:.2f}.")
    if o == "loc":
        return f"Poor box: right class but IoU is only {r['iou']:.2f} with the '{r['gt_label']}' it is trying to cover."
    if o == "both":
        return f"Wrong class and poor box: partly overlaps a '{r['gt_label']}' (IoU {r['iou']:.2f}) and says '{r['label']}'."
    return (f"Background: no ground-truth object under this '{r['label']}' box (best IoU {r['iou']:.2f}) at "
            f"score {r['score']:.2f}. At high scores this is often an unannotated object.")


# ============================================================
# statistics
# ============================================================
def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def one_sided_p(k, n, k_r, n_r):
    """P(slice rate >= observed | same underlying rate), pooled two-proportion z."""
    if n == 0 or n_r == 0:
        return 1.0
    p = (k + k_r) / (n + n_r)
    se = math.sqrt(max(p * (1 - p) * (1 / n + 1 / n_r), 1e-12))
    z = (k / n - k_r / n_r) / se
    return 0.5 * math.erfc(z / math.sqrt(2))


def bh_qvalues(p):
    p = np.asarray(p, float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    q = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def knn_sparsity(emb, ref=None, k=10):
    """Mean cosine distance to the k nearest neighbours. ref=None -> neighbours are the
    other items of emb itself."""
    q = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
    selfref = ref is None
    r = q if selfref else ref / np.maximum(np.linalg.norm(ref, axis=1, keepdims=True), 1e-9)
    q, r = q.astype(np.float32), r.astype(np.float32)
    k = min(k, len(r) - (1 if selfref else 0))
    out = np.zeros(len(q))
    step = max(1, int(2.5e7 // max(len(r), 1)))
    for a in range(0, len(q), step):
        s = q[a:a + step] @ r.T
        if selfref:
            s[np.arange(s.shape[0]), np.arange(a, a + s.shape[0])] = -2
        top = -np.partition(-s, k - 1, axis=1)[:, :k]
        out[a:a + step] = 1 - top.mean(1)
    return out


# ============================================================
# bins, curves, slices
# ============================================================
def add_quartiles(df, cols, defs):
    """Add '<col>_q' string columns (Q1..Q4 with their value ranges) and '<col>_t' tail columns
    (bottom / top 10%, blank in between); defs records the rules. Tails matter because a rare
    condition (e.g. the sparsest 5% of images) is diluted to nothing inside a quartile."""
    for c in cols:
        if c not in df or df[c].notna().sum() < 40:
            continue
        try:
            q = pd.qcut(df[c], 4, duplicates="drop")
        except ValueError:
            continue
        cats = list(q.cat.categories)
        if len(cats) < 2:
            continue
        names = {iv: f"Q{k + 1} ({iv.left:.3g} to {iv.right:.3g})" for k, iv in enumerate(cats)}
        df[c + "_q"] = q.map(names).astype(object)
        defs[c + "_q"] = c
        lo, hi = df[c].quantile([0.1, 0.9])
        tail = pd.Series(np.nan, index=df.index, dtype=object)
        for m, name in ((df[c] <= lo, f"bottom 10% (<= {lo:.3g})"), (df[c] >= hi, f"top 10% (>= {hi:.3g})")):
            if m.mean() <= 0.15:           # skip a "tail" that is really a pile of ties
                tail[m] = name
        if tail.notna().any():
            df[c + "_t"] = tail
            defs[c + "_t"] = c
    return df


def _family(c, qdefs):
    """Slices of one family (e.g. every size bin) yield at most one finding."""
    base = qdefs.get(c, c)
    return FAMILY.get(c) or FAMILY.get(base) or base


def make_curve(df, col, fail, nbins=8):
    v = df[col]
    ok = v.notna().to_numpy()
    if ok.sum() < 80:
        return None
    try:
        b = pd.qcut(v[ok], nbins, duplicates="drop")
    except ValueError:
        return None
    t = pd.DataFrame({"b": b.to_numpy(), "f": fail[ok]}).groupby("b", observed=True)["f"].agg(["sum", "count"])
    if len(t) < 4:
        return None
    bins = []
    for iv, r in t.iterrows():
        lo, hi = wilson(r["sum"], r["count"])
        bins.append(dict(lo=float(iv.left), hi=float(iv.right), n=int(r["count"]),
                         rate=float(r["sum"] / r["count"]), ci=[lo, hi]))
    rates = np.array([x["rate"] for x in bins])
    rng_ = rates.max() - rates.min()
    from scipy.stats import spearmanr
    rho = float(spearmanr(np.arange(len(rates)), rates)[0]) if rng_ > 0 else 0.0
    jumps = np.abs(np.diff(rates))
    jump_share = float(jumps.max() / rng_) if rng_ > 0 else 0.0
    k = int(np.argmax(jumps)) if len(jumps) else 0
    if rng_ < 0.05:
        shape = "flat"
    elif jump_share >= 0.55:
        shape = "cliff"
    elif abs(rho) >= 0.8:
        shape = "smooth"
    else:
        shape = "irregular"
    return dict(col=col, bins=bins, shape=shape, rho=rho, jump_share=jump_share,
                cliff_at=float(bins[k]["hi"]), range=float(rng_),
                direction="rises as it grows" if rho > 0 else "rises as it shrinks")


def enumerate_slices(df, cols, bool_cols, qdefs, min_n):
    out = []
    for c in cols + bool_cols:
        if c not in df:
            continue
        vals = df[c].to_numpy()
        for v in pd.unique(vals[pd.notna(vals)]):
            if c in bool_cols and not v:
                continue
            m = vals == v
            if m.sum() < min_n or (~m).sum() < min_n:
                continue
            base = qdefs.get(c)
            if c in bool_cols:
                label = ATTR_NAMES.get(c, c.replace("issue_", "image issue: ").replace("_", " "))
                definition = f"{c} is true"
            elif base:
                label = f"{ATTR_NAMES.get(base, base)}: {v}"
                definition = f"{base} in {v}"
            else:
                label = f"{ATTR_NAMES.get(c, c)}: {v}"
                definition = f"{c} == {v}"
            out.append(dict(attr=c, value=v if not isinstance(v, (np.bool_,)) else bool(v), label=label,
                            definition=definition, mask=m, family=_family(c, qdefs),
                            ordinal=(c in qdefs or c in ORDINAL_ATTRS)))
    return out


def stat_slices(slices, fail, cfg):
    N, F = len(fail), int(fail.sum())
    for s in slices:
        m = s["mask"]
        n, k = int(m.sum()), int(fail[m].sum())
        n_r, k_r = N - n, F - k
        rate, rate_r = k / n, (k_r / n_r if n_r else 0.0)
        s.update(n=n, k=k, rate=rate, rate_rest=rate_r, ci=list(wilson(k, n)),
                 lift=(rate / rate_r if rate_r > 0 else float("inf")), excess=k - n * rate_r,
                 share_of_failures=(k / F if F else 0.0), p=one_sided_p(k, n, k_r, n_r))
    cand = [s for s in slices if s["k"] >= cfg["min_fail"]]
    q = bh_qvalues([s["p"] for s in cand])
    for s, qq in zip(cand, q):
        s["q"] = float(qq)
    return cand


def select_sequential(cand, fail, cfg):
    """Greedy selection with explaining-away. After each pick, every remaining slice is
    re-tested against the failure rate EXPECTED given membership of the slices already
    chosen (strata = all combinations of chosen slices). A slice survives only if it adds
    failures beyond what the earlier findings already explain; BH is applied per step."""
    N = len(fail)
    ff = fail.astype(float)
    r0 = fail.mean()
    codes = np.zeros(N, np.int64)
    total_fail = max(int(fail.sum()), 1)
    pool = [s for s in cand if s["q"] < cfg["fdr"] and s["excess"] > 0 and
            (s["lift"] >= cfg["min_lift"] or s["rate"] - s["rate_rest"] >= cfg["min_diff"])]
    chosen = []
    while pool and len(chosen) < cfg["top_k"]:
        G = int(codes.max()) + 1
        res = []
        for s in pool:
            m = s["mask"]
            out = ~m
            cnt = np.bincount(codes[out], minlength=G).astype(float)
            fl = np.bincount(codes[out], weights=ff[out], minlength=G)
            r_u = ((fl + 5.0 * r0) / (cnt + 5.0))[codes[m]]
            exp, var = r_u.sum(), (r_u * (1 - r_u)).sum()
            z = (s["k"] - exp - 0.5) / math.sqrt(max(var, 1e-9))
            res.append((s, exp, 0.5 * math.erfc(z / math.sqrt(2))))
        qs = bh_qvalues([r[2] for r in res])
        ok = [(r, q) for r, q in zip(res, qs)
              if q < cfg["fdr"] and r[0]["k"] - r[1] >= cfg["min_share"] * total_fail and
              (r[0]["k"] / max(r[1], 1e-9) >= cfg["min_adj_lift"] or (r[0]["k"] - r[1]) / r[0]["n"] >= cfg["min_diff"])]
        if not ok:
            break
        (s, exp, _), q = max(ok, key=lambda t: t[0][0]["k"] - t[0][1])
        # prefer the tighter description: a slice mostly inside the pick that carries most of
        # its excess failures (the rest of the pick adds at most half as much again) names the
        # problem better ("dark images" over "low contrast")
        tighter = [(r, qq) for r, qq in ok if r[0]["n"] < s["n"]
                   and (r[0]["mask"] & s["mask"]).sum() >= 0.8 * r[0]["n"]
                   and r[0]["k"] - r[1] >= (2 / 3) * (s["k"] - exp)]
        if tighter:
            broader = s
            (s, exp, _), q = min(tighter, key=lambda t: t[0][0]["n"])
            s.setdefault("aliases", []).append(broader["label"])
            pool = [x for x in pool if x is not broader]
        s.update(adj_expected=float(exp), adj_excess=float(s["k"] - exp), adj_lift=float(s["k"] / exp),
                 adj_q=float(q), after=[c["label"] for c in chosen])
        chosen.append(s)
        # near-identical slices from other attributes are the same finding under another name
        same = [x for x in pool if x is not s and
                (x["mask"] & s["mask"]).sum() / (x["n"] + s["n"] - (x["mask"] & s["mask"]).sum()) >= 0.7]
        s.setdefault("aliases", []).extend(x["label"] for x in same)
        pool = [x for x in pool if not any(x is y for y in same)]
        if s["ordinal"]:      # one finding per ordinal family; its other bins live in the dose-response curve
            for x in pool:    # (only bins that mostly overlap this one are the same finding by another name)
                if x is not s and x["ordinal"] and x["family"] == s["family"] and \
                        (x["mask"] & s["mask"]).sum() >= 0.6 * min(x["n"], s["n"]):
                    s.setdefault("aliases", []).append(x["label"])
            pool = [x for x in pool if not (x["ordinal"] and x["family"] == s["family"])]
        else:
            pool = [x for x in pool if x is not s]
        codes = codes * 2 + s["mask"].astype(np.int64)
        _, codes = np.unique(codes, return_inverse=True)
    return chosen


def mark_refinements(chosen):
    for i, s in enumerate(chosen):
        s["refines"] = None
        for c in chosen[:i]:
            if c["mask"].sum() > s["mask"].sum() and (s["mask"] & c["mask"]).sum() / s["mask"].sum() >= 0.8:
                s["refines"] = c["label"]
                break


def attach_aliases(cand, chosen, cfg):
    """Unchosen but raw-significant slices that mostly overlap a chosen one are shown as
    'also seen as' on that finding."""
    ids = {id(c) for c in chosen}
    for s in cand:
        if id(s) in ids or s.get("q", 1) >= cfg["fdr"] or not (
                s["lift"] >= cfg["min_lift"] or s["rate"] - s["rate_rest"] >= cfg["min_diff"]):
            continue
        best, bo = None, 0.0
        for c in chosen:
            o = (s["mask"] & c["mask"]).sum() / max(min(s["mask"].sum(), c["mask"].sum()), 1)
            if o > bo:
                best, bo = c, o
        if best is not None and bo >= 0.6:
            best.setdefault("aliases", []).append(s["label"])


# ============================================================
# cause battery
# ============================================================
VERDICT_RANK = {"supports": 2, "partial": 1, "against": 0, "n/a": -1}


def _h(key, title, verdict, evidence, action="", **numbers):
    return dict(key=key, title=title, verdict=verdict, evidence=evidence, action=action, numbers=numbers)


def _pct(x):
    return f"{x:.0%}"


class Ctx:
    pass


def _confounding(ctx, s, cols):
    df, m = ctx.df, s["mask"]
    rest = ~m
    rate_r = ctx.fail[rest].mean()
    excess = s["k"] - s["n"] * rate_r
    if excess <= 0:
        return None
    best = None
    for c in cols:
        if c not in df or _family(c, ctx.qdefs) == s["family"]:
            continue
        vals = df[c].to_numpy()
        r = pd.DataFrame({"v": vals[rest], "f": ctx.fail[rest]}).groupby("v")["f"].agg(["mean", "count"])
        r = r[r["count"] >= 20]["mean"]
        exp = pd.Series(vals[m]).map(r).fillna(rate_r).sum()
        explained = float(np.clip((exp - s["n"] * rate_r) / excess, 0, 1))
        if best is None or explained > best[1]:
            best = (c, explained, float(exp))
    return best


def _curve_for(ctx, s):
    fam = s["family"]
    col = NUM_FAMILY_COL.get(fam, fam)
    return ctx.curves.get(col)


def battery(ctx, s):
    """Hypotheses for one slice. ctx.side is 'recall' (ground-truth objects that were not
    found) or 'precision' (predictions that are wrong)."""
    cfg, df, m = ctx.cfg, ctx.df, s["mask"]
    sub = df[m]
    f = ctx.fail[m]
    failed = sub[f]
    nf = len(failed)
    out = []
    oc = failed["outcome"].value_counts().to_dict()
    thr = cfg["conf_thresh"]

    # --- label problems (recall side only: needs suspect flags)
    if ctx.side == "recall":
        if not ctx.has_suspect:
            out.append(_h("label_issue", "Labels in this slice may be wrong", "n/a",
                          "No suspect-label flags were supplied, so this was not tested."))
        else:
            sus = sub["suspect"].to_numpy(bool)
            sf = int((sus & f).sum())
            ns = ~sus
            rate_ns = f[ns].mean() if ns.any() else 0
            rest_ns = ctx.fail[(~m) & ~ctx.suspect]
            rate_r_ns = rest_ns.mean() if len(rest_ns) else 0
            lift_ns = rate_ns / rate_r_ns if rate_r_ns > 0 else float("inf")
            ev = (f"{sf} of {nf} failures ({_pct(sf / max(nf, 1))}) are on boxes flagged as suspect labels. "
                  f"Excluding flagged boxes the failure rate goes {_pct(s['rate'])} to {_pct(rate_ns)} "
                  f"(lift {s['lift']:.1f}x to {lift_ns:.1f}x).")
            if sf and "suspect_reason" in sub:
                why = sub.loc[sus & f, "suspect_reason"].fillna("").replace("", "unspecified").value_counts()
                words = {"class_disagree": "an independent model says another class (two models agree it is "
                                           "not this class: a label error or a confusion they share)",
                         "spurious": "an independent model sees no object there at all"}
                ev += " Flag reasons: " + "; ".join(f"{n} {words.get(r, r)}" for r, n in why.items()) + "."
            v = ("supports" if lift_ns < 1.15 and sf > 0 else
                 "partial" if sf / max(nf, 1) >= 0.25 else "against")
            out.append(_h("label_issue", "Labels in this slice may be wrong", v, ev,
                          "Review these boxes in the label pipeline before treating this as a model weakness.",
                          suspect_failures=sf, lift_without_suspects=float(lift_ns)))

    # --- confounding
    cols = ctx.confounders
    cf = _confounding(ctx, s, cols)
    if cf is None:
        out.append(_h("confounded", "Explained by another attribute", "n/a", "The slice has no excess to explain."))
    else:
        c, ex, exp = cf
        base = ctx.qdefs.get(c, c)
        nm = ATTR_NAMES.get(base, base)
        v = "supports" if ex >= 0.6 else "partial" if ex >= 0.3 else "against"
        if v == "against":
            ev = (f"The excess survives matching on class, size, overlap, crowding and scene group "
                  f"(best single explainer: {nm}, {_pct(ex)}).")
        else:
            ev = (f"Matching on {nm} alone, this slice would be expected to fail {exp:.0f} times instead of "
                  f"{s['k']}: {nm} explains {_pct(ex)} of the excess.")
        out.append(_h("confounded", f"Largely a proxy for {nm}" if v != "against" else "Explained by another attribute",
                      v, ev, f"Look at {nm} first; improving it should lift this slice too." if v != "against" else "",
                      explained=ex, by=c))

    if ctx.side == "recall":
        miss = failed[failed["outcome"] == "missed"]
        nm_ = len(miss)
        # --- operating point
        low = miss[miss["why"] == "low_score_correct"]
        if nm_ >= 10:
            share = len(low) / nm_
            med = float(low["pred_score"].median()) if len(low) else float("nan")
            rec = int((miss["best_score_same"] >= 0.3).sum())
            v = "supports" if share >= 0.35 else "partial" if share >= 0.2 else "against"
            ev = (f"{len(low)} of {nm_} misses ({_pct(share)}) have a correct-class box scored under {thr}"
                  + (f" (median {med:.2f})" if len(low) else "") +
                  f". At a 0.3 threshold about {rec} of the {nf} failures here would be recovered, at some cost in false positives.")
            out.append(_h("threshold_effect", "The model finds these but scores them too low", v, ev,
                          "Tune the threshold per class or slice, or recalibrate scores; check the precision side for the cost.",
                          share=share, recoverable_at_0_3=rec))
        else:
            out.append(_h("threshold_effect", "The model finds these but scores them too low", "n/a",
                          f"Only {nm_} misses in this slice; too few to judge."))
        # --- never proposed
        if nm_ >= 10:
            npz = int((miss["why"] == "no_proposal").sum())
            share = npz / nm_
            v = "supports" if share >= 0.5 else "partial" if share >= 0.3 else "against"
            out.append(_h("no_proposal", "The model never proposes these objects", v,
                          f"{npz} of {nm_} misses ({_pct(share)}) have no overlapping prediction at any score.",
                          "A recall ceiling: raise input resolution or use finer anchors for small objects, or add training data.",
                          share=share))
            sh = int((miss["why"] == "shadowed").sum())
            share = sh / nm_
            v = "supports" if share >= 0.25 else "partial" if share >= 0.12 else "against"
            out.append(_h("crowd_suppression", "Suppressed by a neighbouring detection (NMS)", v,
                          f"{sh} of {nm_} misses ({_pct(share)}) are overlapped by a same-class detection that matched a neighbour. "
                          "This is a heuristic; the model's pre-NMS output is not available.",
                          "Try Soft-NMS or a higher NMS IoU threshold, or a crowd-aware head.", share=share))
        # --- confusion
        nc = oc.get("confused", 0)
        if nc >= 8:
            conf = failed[failed["outcome"] == "confused"]
            med = float(conf["pred_score"].median())
            pairs = Counter(zip(conf["label"], conf["pred_label"])).most_common(3)
            pstr = ", ".join(f"{a} as {b} ({n})" for (a, b), n in pairs)
            share = nc / max(nf, 1)
            if share >= 0.25:
                v = "supports" if med >= 0.7 else "partial"
                ev = (f"{nc} failures ({_pct(share)}) are correct locations with the wrong class; median wrong-class score "
                      f"{med:.2f}. Most common: {pstr}.")
                act = ("Confidently wrong: check for ambiguous class definitions or inconsistent labels, then add contrasting examples."
                       if med >= 0.7 else "Uncertain confusion: add examples of the confused classes and hard negatives.")
            else:
                v, ev, act = "against", f"Confusions are only {_pct(share)} of failures here. Most common: {pstr}.", ""
            out.append(_h("confusion", "Classes are being confused", v, ev, act, share=share, median_wrong_score=med))
        else:
            out.append(_h("confusion", "Classes are being confused", "against" if nf >= 10 else "n/a",
                          f"Only {nc} wrong-class failures here."))
        # --- localization
        nl = oc.get("mislocalized", 0)
        share = nl / max(nf, 1)
        if nf >= 10:
            v = "supports" if share >= 0.3 else "partial" if share >= 0.15 else "against"
            med = float(failed.loc[failed["outcome"] == "mislocalized", "iou"].median()) if nl else float("nan")
            out.append(_h("localization", "Boxes are found but badly aligned", v,
                          f"{nl} failures ({_pct(share)}) are right-class boxes with IoU below {cfg['iou_fg']}"
                          + (f" (median IoU {med:.2f})." if nl else "."),
                          "Check the box convention in the labels; consider higher resolution or an IoU-aware box loss.",
                          share=share))
    else:
        npf = len(failed)
        if npf >= 10:
            nb = oc.get("bkg", 0)
            share = nb / npf
            med = float(failed.loc[failed["outcome"] == "bkg", "score"].median()) if nb else float("nan")
            hi = int((failed[failed["outcome"] == "bkg"]["score"] >= 0.8).sum())
            # background hits split by score: confident ones are often real but unlabelled
            # objects, moderate ones are the model firing on background patterns
            v = ("supports" if share >= 0.5 and med >= 0.8 else
                 "partial" if share >= 0.3 and hi / npf >= 0.2 else "against")
            out.append(_h("unannotated", "Many of these may be real objects missing from the labels", v,
                          f"{nb} of {npf} false positives ({_pct(share)}) sit on pure background"
                          + (f", median score {med:.2f}; {hi} score at least 0.8." if nb else "."),
                          "Send the high-scoring ones to the label pipeline: confident background hits are often unannotated objects.",
                          share=share))
            v = ("supports" if share >= 0.5 and med < 0.8 else
                 "partial" if share >= 0.3 and (nb - hi) / npf >= 0.2 else "against")
            out.append(_h("background_fp", "The model fires on background", v,
                          f"{nb} of {npf} false positives ({_pct(share)}) have no object under them"
                          + (f"; {nb - hi} of those score under 0.8 (median {med:.2f}), too low to be confident "
                             "unlabelled objects." if nb else "."),
                          "Raise the threshold here (see the threshold sweep for the recall cost) or add hard "
                          "negatives that look like this slice.", share=share))
            nd = oc.get("dupe", 0)
            share = nd / npf
            v = "supports" if share >= 0.25 else "partial" if share >= 0.12 else "against"
            out.append(_h("duplicates", "Duplicate boxes survive NMS", v,
                          f"{nd} of {npf} false positives ({_pct(share)}) are second boxes on an already-detected object.",
                          "Tighten the NMS IoU threshold, or use class-agnostic NMS.", share=share))
            nc = oc.get("cls", 0) + oc.get("both", 0)
            share = nc / npf
            conf = failed[failed["outcome"].isin(["cls", "both"])]
            med = float(conf["score"].median()) if nc else float("nan")
            pairs = Counter(zip(conf["gt_label"], conf["label"])).most_common(3)
            v = "supports" if share >= 0.25 and med >= 0.7 else "partial" if share >= 0.25 else "against"
            out.append(_h("confusion", "Classes are being confused", v,
                          f"{nc} false positives ({_pct(share)}) are in the right place with the wrong class"
                          + (f"; most common: " + ", ".join(f"{a} as {b} ({n})" for (a, b), n in pairs) if nc else "."),
                          "Add contrasting examples; check whether the confused classes are labelled consistently.", share=share))
            nl = oc.get("loc", 0)
            share = nl / npf
            v = "supports" if share >= 0.3 else "partial" if share >= 0.15 else "against"
            out.append(_h("localization", "Boxes are found but badly aligned", v,
                          f"{nl} false positives ({_pct(share)}) are right-class boxes with IoU below {cfg['iou_fg']}.",
                          "Check the box convention in the labels; consider higher resolution or an IoU-aware box loss.", share=share))

    # --- data coverage (proxy unless a training reference was supplied)
    if ctx.img_pct is None:
        out.append(_h("data_sparsity", "Few similar examples in the data", "n/a", "No embeddings were supplied."))
    else:
        imgs = np.unique(ctx.unit_img[m])
        if len(imgs) < 10 or len(imgs) > 0.7 * ctx.n_img:
            out.append(_h("data_sparsity", "Few similar examples in the data", "n/a",
                          "The slice covers too few (or almost all) images to compare density."))
        else:
            obs = float(ctx.img_pct[imgs].mean())
            draws = ctx.img_pct[ctx.rng.random((300, ctx.n_img)).argsort(1)[:, :len(imgs)]].mean(1)
            p = float((draws >= obs).mean())
            v = "supports" if obs >= 0.6 and p < 0.05 else "partial" if obs >= 0.55 and p < 0.1 else "against"
            out.append(_h("data_sparsity", "Few similar examples in the data", v,
                          f"These images sit at the {obs:.0%} point of CLIP-space sparsity on average (50% is typical; "
                          f"permutation p={p:.3f}). Reference: {ctx.ref_name}.",
                          "Collect or synthesize more examples that look like this slice.", mean_pct=obs, p=p))

    # --- dose-response
    cv = _curve_for(ctx, s)
    if cv is None:
        out.append(_h("intrinsic", "Failure grows steadily with a measurable difficulty", "n/a",
                      "This slice is not defined by a measurable quantity, so there is no dose-response curve."))
    else:
        nm = ATTR_NAMES.get(cv["col"], cv["col"])
        lo, hi = cv["bins"][0]["rate"], cv["bins"][-1]["rate"]
        if cv["shape"] == "smooth":
            out.append(_h("intrinsic", "Failure grows steadily with a measurable difficulty", "supports",
                          f"Failure rate {cv['direction']} with {nm}: {_pct(lo)} to {_pct(hi)} across 8 bins (rank correlation {cv['rho']:+.2f}). A smooth ramp, no single break.",
                          f"Expected difficulty at the hard end of {nm}; target it with resolution, augmentation or data.",
                          rho=cv["rho"]))
        elif cv["shape"] == "cliff":
            out.append(_h("intrinsic", "A sharp break in failure rate", "supports",
                          f"Most of the change in failure rate ({_pct(cv['jump_share'])}) happens at one step near {nm} = {cv['cliff_at']:.3g}.",
                          "A sharp break usually points to an architectural limit (stride, input size, anchor scale).",
                          cliff_at=cv["cliff_at"]))
        elif cv["shape"] == "irregular":
            out.append(_h("intrinsic", "Failure grows steadily with a measurable difficulty", "partial",
                          f"Failure varies with {nm} ({_pct(lo)} to {_pct(hi)}) but not monotonically.", ""))
        else:
            out.append(_h("intrinsic", "Failure grows steadily with a measurable difficulty", "against",
                          f"Failure rate is flat across {nm}.", ""))
    return out


def rank_hypotheses(hyps):
    order = ["label_issue", "confounded", "unannotated", "background_fp", "threshold_effect", "confusion", "duplicates",
             "localization", "crowd_suppression", "no_proposal", "data_sparsity", "intrinsic"]
    return sorted(hyps, key=lambda h: (-VERDICT_RANK[h["verdict"]], order.index(h["key"]) if h["key"] in order else 99))


def headline(s, hyps, side):
    noun = "of these objects are not found" if side == "recall" else "of these detections are wrong"
    base = (f"{_pct(s['rate'])} {noun}, versus {_pct(s['rate_rest'])} elsewhere ({s['lift']:.1f}x). "
            f"They make up {_pct(s['share_all'])} of all failures.")
    if s["after"]:
        base += (f" Even after accounting for the findings above, about {s['adj_excess']:.0f} more fail than "
                 f"expected ({s['adj_lift']:.1f}x).")
    top = [h for h in hyps if h["verdict"] == "supports"][:2]
    if top:
        base += " Best supported: " + "; ".join(h["title"][0].lower() + h["title"][1:] for h in top) + "."
    elif any(h["verdict"] == "partial" for h in hyps):
        base += " Evidence is partial: " + hyps[0]["title"][0].lower() + hyps[0]["title"][1:] + "."
    else:
        base += " None of the tests explains it, so this is inconclusive."
    return base


# ============================================================
# findings
# ============================================================
def _examples(ctx, s, k=12):
    sub = ctx.df[s["mask"] & ctx.fail]
    if len(sub) == 0:
        return []
    sub = sub.sample(frac=1, random_state=ctx.cfg["seed"]).drop_duplicates("image_id").head(k)
    cols = ["image_id", ctx.idx_col, "label", "outcome", "x", "y", "w", "h"]
    if "why" in sub:
        cols.append("why")
    return json.loads(sub[cols].rename(columns={ctx.idx_col: "idx"}).to_json(orient="records"))


def build_cards(ctx, chosen, total_fail):
    """(cards, members): members lists every unit in each finding's slice, failed or not."""
    cards, members = [], []
    for s in chosen:
        s["share_all"] = s["k"] / max(total_fail, 1)
        hyps = rank_hypotheses(battery(ctx, s))
        sub = ctx.df[s["mask"] & ctx.fail]
        fid = f"{ctx.side}:{s['attr']}={s['value']}"
        members.append(pd.DataFrame({"finding_id": fid, "side": ctx.side,
                                     "image_id": ctx.df.loc[s["mask"], "image_id"].to_numpy(),
                                     "idx": ctx.df.loc[s["mask"], ctx.idx_col].to_numpy(),
                                     "failed": ctx.fail[s["mask"]]}))
        cards.append(dict(
            id=fid, side=ctx.side, label=s["label"], definition=s["definition"],
            aliases=list(dict.fromkeys(s.get("aliases", [])))[:6], n=s["n"], failures=s["k"], rate=s["rate"], ci=s["ci"],
            rate_rest=s["rate_rest"], lift=None if math.isinf(s["lift"]) else s["lift"], excess=s["excess"],
            share_of_failures=s["share_all"], q=s["q"], after=s["after"], refines=s.get("refines"), adj_expected=s["adj_expected"],
            adj_excess=s["adj_excess"], adj_lift=s["adj_lift"], adj_q=s["adj_q"],
            modes={k: int(v) for k, v in sub["outcome"].value_counts().items()},
            reasons={k: int(v) for k, v in sub["why"].value_counts().items()} if "why" in sub else {},
            headline=headline(s, hyps, ctx.side),
            hypotheses=hyps, examples=_examples(ctx, s), not_run=NOT_RUN))
    return cards, members


def mine(side, df, fail, cfg, cat, bool_, qdefs, curves, confounders, img_pct, unit_img, n_img, ref_name, suspect):
    df = df.reset_index(drop=True)
    cand = stat_slices(enumerate_slices(df, cat, bool_, qdefs, cfg["min_slice"]), fail, cfg)
    chosen = select_sequential(cand, fail, cfg)
    attach_aliases(cand, chosen, cfg)
    mark_refinements(chosen)
    ctx = make_ctx(side, df, fail, cfg, curves, qdefs, confounders, img_pct, unit_img, n_img, ref_name, suspect)
    return build_cards(ctx, chosen, int(fail.sum()))   # (cards, members)


def make_ctx(side, df, fail, cfg, curves, qdefs, confounders, img_pct, unit_img, n_img, ref_name, suspect):
    c = Ctx()
    c.side, c.df, c.fail, c.cfg, c.curves, c.qdefs = side, df.reset_index(drop=True), fail, cfg, curves, qdefs
    c.confounders, c.img_pct, c.unit_img, c.n_img, c.ref_name = confounders, img_pct, unit_img, n_img, ref_name
    c.suspect = suspect
    c.has_suspect = bool(suspect.any())
    c.rng = np.random.default_rng(cfg["seed"])
    c.idx_col = "gt_idx" if side == "recall" else "pred_idx"
    return c


# ============================================================
# scene clusters
# ============================================================
def discover_clusters(emb, ids, img_df, gt_units, k=None, seed=0):
    from sklearn.cluster import KMeans
    n = len(ids)
    if n < 60:
        return None, []
    k = k or int(np.clip(n // 100, 6, 40))
    x = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
    km = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(x)
    lab = km.labels_
    order = np.argsort(np.bincount(lab))[::-1]          # C01 is the biggest group
    rename = {int(o): f"C{i + 1:02d}" for i, o in enumerate(order)}
    names = np.array([rename[int(l)] for l in lab])
    cls_all = gt_units[~gt_units["crowd"]]["label"].value_counts(normalize=True)
    per = gt_units[~gt_units["crowd"]].merge(pd.DataFrame({"image_id": ids, "cl": names}), on="image_id")
    info = []
    gstd = {c: img_df[c].std() for c in CLUSTER_WORDS if c in img_df}
    for c in sorted(set(names)):
        idx = np.flatnonzero(names == c)
        sub = per[per["cl"] == c]
        top = []
        if len(sub):
            f = sub["label"].value_counts()
            lift = (f / len(sub)) / cls_all.reindex(f.index)
            top = [l for l in lift[f >= 5].sort_values(ascending=False).index[:2]]
        words = []
        for col, (lo_w, hi_w) in CLUSTER_WORDS.items():
            if col in img_df and gstd.get(col, 0) > 0:
                z = (img_df[col].to_numpy()[idx].mean() - img_df[col].mean()) / gstd[col]
                if abs(z) >= 0.6:
                    words.append((abs(z), hi_w if z > 0 else lo_w))
        nobj = sub.groupby("image_id").size().reindex(ids[idx]).fillna(0).mean() if len(sub) else 0
        allobj = per.groupby("image_id").size().mean() if len(per) else 0
        if allobj and nobj > 1.5 * allobj:
            words.append((1.0, "crowded"))
        words = [w for _, w in sorted(words, reverse=True)[:2]]
        desc = ", ".join(top) if top else "mixed"
        if words:
            desc += f" ({', '.join(words)})"
        cen = km.cluster_centers_[[k_ for k_, v in rename.items() if v == c][0]]
        cen = cen / max(np.linalg.norm(cen), 1e-9)
        rep = idx[np.argsort(-(x[idx] @ cen))[:6]]
        info.append(dict(id=c, label=desc, n_images=int(len(idx)), rep_images=[str(ids[i]) for i in rep]))
    return names, info


# ============================================================
# orchestrator
# ============================================================
def _load(adir):
    p = lambda n: os.path.join(adir, n)
    images = pd.read_csv(p("images.csv"), dtype={"image_id": str})
    gt = pd.read_csv(p("gt.csv"), dtype={"image_id": str})
    preds = pd.read_csv(p("preds.csv"), dtype={"image_id": str})
    for c, d in (("iscrowd", 0), ("suspect", 0)):
        if c not in gt:
            gt[c] = d
        gt[c] = gt[c].fillna(0).astype(int).astype(bool)
    emb = ids = pts = ref = None
    if os.path.isfile(p("embeddings.npz")):
        z = np.load(p("embeddings.npz"), allow_pickle=False)
        ids, emb = z["image_ids"].astype(str), z["emb"].astype(np.float32)
    if os.path.isfile(p("points2d.npz")):
        z = np.load(p("points2d.npz"), allow_pickle=False)
        pts = dict(zip(z["image_ids"].astype(str), z["xy"].tolist()))
    if os.path.isfile(p("reference_emb.npz")):
        ref = np.load(p("reference_emb.npz"), allow_pickle=False)["emb"].astype(np.float32)
    return images, gt, preds, ids, emb, pts, ref


def run(adir, **kw):
    t0 = time.time()
    cfg = {**CFG, **kw}
    images, gt, preds, emb_ids, emb, pts, ref = _load(adir)
    print(f"Loaded {len(images)} images, {len(gt)} GT boxes, {len(preds)} predictions "
          f"(operating point: score >= {cfg['conf_thresh']}, IoU >= {cfg['iou_fg']})")

    # ---- image-level attributes
    img = images.set_index("image_id", drop=False)
    img.index.name = None
    if "issues" in img:
        iss = img["issues"].fillna("").astype(str)
        for name in sorted({t for s in iss for t in s.split(";") if t}):
            img["issue_" + name] = iss.str.contains(rf"(?:^|;){name}(?:;|$)", regex=True)
    num = [c for c in IMG_NUM if c in img and img[c].notna().sum() > 40]
    boo = [c for c in IMG_BOOL + [c for c in img if c.startswith("issue_")] if c in img]
    img_pct = sparsity = None
    ref_name = "none"
    scene = None
    clusters = []
    if emb is not None:
        pos = {i: k for k, i in enumerate(emb_ids)}
        keep = [i for i in img.index if i in pos]
        e = emb[[pos[i] for i in keep]]
        if ref is not None:
            sp = knn_sparsity(e, ref, cfg["knn"])
            ref_name = f"training set ({len(ref)} images)"
        else:
            sp = knn_sparsity(e, None, cfg["knn"])
            ref_name = "this evaluation set only (a proxy: supply reference_emb.npz for training coverage)"
        from scipy.stats import rankdata
        img["sparsity"] = pd.Series(sp, index=keep)
        img_pct = (rankdata(img["sparsity"].fillna(img["sparsity"].median())) / len(img))
        num.append("sparsity")

    # ---- matching
    g_by = {k: v for k, v in gt.groupby("image_id")}
    p_by = {k: v.sort_values("confidence", ascending=False, kind="stable") for k, v in preds.groupby("image_id")}
    empty_g = gt.iloc[:0]
    empty_p = preds.iloc[:0]
    G, P, S = [], [], []
    for iid, W, H in zip(img["image_id"], img["width"], img["height"]):
        a, b, c = analyze_image(W, H, g_by.get(iid, empty_g), p_by.get(iid, empty_p), cfg)
        G += a
        P += b
        S += c
    gu, pu, su = pd.DataFrame(G), pd.DataFrame(P), pd.DataFrame(S)
    print(f"Matched in {time.time() - t0:.1f}s")

    # ---- threshold sweep (counts only)
    sweep = []
    for t in cfg["sweep"]:
        c2 = {**cfg, "conf_thresh": t}
        tot = Counter()
        for iid, W, H in zip(img["image_id"], img["width"], img["height"]):
            tot.update(analyze_image(W, H, g_by.get(iid, empty_g), p_by.get(iid, empty_p), c2, light=True))
        fp = sum(tot[k] for k in ("dupe", "cls", "loc", "both", "bkg"))
        sweep.append(dict(thresh=t, recall=tot["tp"] / max(tot["n_gt"], 1),
                          precision=tot["tp"] / max(tot["tp"] + fp, 1), tp=tot["tp"], fp=fp,
                          **{k: tot[k] for k in ("dupe", "cls", "loc", "both", "bkg")}))

    # ---- join image attributes + scene groups
    keep_cols = ["image_id"] + [c for c in num + boo if c in img]
    ia = img[keep_cols].copy()
    if emb is not None and len(keep):
        names, clusters = discover_clusters(e, np.array(keep), img.loc[keep], gu, cfg["n_clusters"], cfg["seed"])
        if names is not None:
            ia["scene_cluster"] = pd.Series(names, index=keep)
    gu = gu.merge(ia, on="image_id", how="left")
    pu = pu.merge(ia, on="image_id", how="left")
    gu["explain"] = [explain_gt(r, cfg) for r in gu.to_dict("records")]
    pu["explain"] = [explain_pred(r, cfg) for r in pu.to_dict("records")]
    img_index = {i: k for k, i in enumerate(img["image_id"])}

    # ---- recall side
    sc = gu[~gu["crowd"]].reset_index(drop=True)
    fail_r = (sc["outcome"] != "detected").to_numpy()
    qdefs = {}
    qcols_g = ["area_px", "overlap_max", "center_dist"] + [c for c in num]
    sc = add_quartiles(sc, qcols_g, qdefs)
    cat_g = ["label", "size_bucket", "aspect_bucket", "overlap_bucket", "n_obj_bucket", "position",
             "scene_cluster"] + [c for c in sc if c.endswith(("_q", "_t"))]
    bool_g = ["truncated", "suspect"] + [c for c in boo if c in sc]
    curves_g = {c: make_curve(sc, c, fail_r) for c in ["area_px", "overlap_max", "n_objects", "center_dist"] + num}
    curves_g = {k: v for k, v in curves_g.items() if v}
    conf_g = ["label", "size_bucket", "overlap_bucket", "n_obj_bucket", "scene_cluster", "brightness_q"]
    cards_g, members_g = mine("recall", sc, fail_r, cfg, cat_g, bool_g, qdefs, curves_g, conf_g, img_pct,
                              sc["image_id"].map(img_index).to_numpy(), len(img), ref_name,
                              sc["suspect"].to_numpy(bool))

    # ---- precision side
    pc = pu.reset_index(drop=True)
    fail_p = (pc["outcome"] != "tp").to_numpy()
    qdefs_p = {}
    pc = add_quartiles(pc, ["area_px", "center_dist", "score"] + num, qdefs_p)
    cat_p = ["label", "size_bucket", "aspect_bucket", "n_preds_bucket", "position", "scene_cluster"] + \
            [c for c in pc if c.endswith(("_q", "_t")) and not c.startswith("score_")]   # score is a curve, not a slice
    bool_p = ["truncated"] + [c for c in boo if c in pc]
    curves_p = {c: make_curve(pc, c, fail_p) for c in ["area_px", "score", "center_dist"] + num}
    curves_p = {k: v for k, v in curves_p.items() if v}
    cards_p, members_p = mine("precision", pc, fail_p, cfg, cat_p, bool_p, qdefs_p, curves_p,
                              ["label", "size_bucket", "n_preds_bucket", "scene_cluster", "brightness_q"], img_pct,
                              pc["image_id"].map(img_index).to_numpy(), len(img), ref_name, np.zeros(len(pc), bool))

    # ---- overview
    n_gt = len(sc)
    oc = sc["outcome"].value_counts().to_dict()
    miss_why = sc[sc["outcome"] == "missed"]["why"].value_counts().to_dict()
    fp_types = pc[pc["outcome"] != "tp"]["outcome"].value_counts().to_dict()
    cls_rows = []
    for lab, grp in sc.groupby("label"):
        pr = pc[pc["label"] == lab]
        top_fail = grp[grp["outcome"] != "detected"]["outcome"].value_counts()
        cls_rows.append(dict(label=lab, n_gt=len(grp), recall=float((grp["outcome"] == "detected").mean()),
                             n_pred=len(pr), precision=float((pr["outcome"] == "tp").mean()) if len(pr) else None,
                             main_failure=(top_fail.index[0] if len(top_fail) else None)))
    cls_rows.sort(key=lambda r: r["recall"])
    confusions = Counter(zip(sc[sc["outcome"] == "confused"]["label"], sc[sc["outcome"] == "confused"]["pred_label"]))
    sus = sc["suspect"].to_numpy(bool)
    overview = dict(
        config={k: cfg[k] for k in ("conf_thresh", "iou_fg", "iou_bg", "shadow_iou")},
        counts=dict(images=len(img), gt=n_gt, crowd_ignored=int(gu["crowd"].sum()), predictions_at_op=len(pc),
                    predictions_total=int(len(preds))),
        recall=float(oc.get("detected", 0) / max(n_gt, 1)),
        precision=float((pc["outcome"] == "tp").mean()) if len(pc) else None,
        gt_outcomes={k: int(v) for k, v in oc.items()}, miss_reasons={k: int(v) for k, v in miss_why.items()},
        fp_types={k: int(v) for k, v in fp_types.items()}, tp=int((pc["outcome"] == "tp").sum()),
        sweep=sweep, classes=cls_rows,
        confusions=[dict(gt=a, pred=b, n=int(n)) for (a, b), n in confusions.most_common(15)],
        curves_recall={k: v for k, v in curves_g.items()}, curves_precision={k: v for k, v in curves_p.items()},
        suspect=dict(n=int(sus.sum()),
                     recall_without=float((sc["outcome"][~sus] == "detected").mean()) if sus.any() else None,
                     failure_rate_flagged=float(fail_r[sus].mean()) if sus.any() else None),
        reference=ref_name, has_embeddings=emb is not None)

    caveats = ["Slices overlap and the evidence is observational: a supported hypothesis is a lead to confirm, not proof of cause.",
               "Counts are at one operating point and match FiftyOne's evaluate_detections box for box, except that "
               "predictions inside crowd regions are ignored here (COCO convention) while FiftyOne counts them as "
               "true positives, so its Model Evaluation panel shows a higher precision.",
               "Many slices were tested; p-values were corrected for that (Benjamini-Hochberg), but small slices can still mislead."]
    if emb is not None and ref is None:
        caveats.append("The data-coverage test compares density within the evaluation set, not against the training set.")
    if not sus.any():
        caveats.append("No suspect-label flags were supplied, so label problems could not be separated from model problems.")

    out = lambda n: os.path.join(adir, n)
    gu.to_csv(out("gt_units.csv"), index=False)
    pd.concat([pu, su], ignore_index=True).to_csv(out("pred_units.csv"), index=False)
    _dump(out("overview.json"), overview)
    _dump(out("findings.json"), dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"), config={k: cfg[k] for k in (
        "conf_thresh", "iou_fg", "iou_bg", "fdr", "min_lift", "min_slice")}, recall=cards_g, precision=cards_p,
        caveats=caveats))
    # every unit in every finding's slice (idx = gt_idx on the recall side, pred_idx on the precision side)
    mem = members_g + members_p
    (pd.concat(mem, ignore_index=True) if mem else
     pd.DataFrame(columns=["finding_id", "side", "image_id", "idx", "failed"])).to_csv(out("finding_members.csv"), index=False)
    # clusters with failure rates
    if clusters:
        fr = sc.groupby("scene_cluster")["outcome"].agg(lambda o: float((o != "detected").mean()))
        for c in clusters:
            c["gt_failure_rate"] = float(fr.get(c["id"], float("nan"))) if c["id"] in fr.index else None
            c["n_gt"] = int((sc["scene_cluster"] == c["id"]).sum())
    _dump(out("clusters.json"), clusters)
    # failure map
    if pts:
        gf = sc.groupby("image_id")["outcome"].agg(lambda s: float((s != "detected").mean()))
        pf = pc.groupby("image_id")["outcome"].agg(lambda s: float((s != "tp").mean()))
        cl = ia.set_index("image_id").get("scene_cluster")
        rows = []
        for iid, xy in pts.items():
            if iid in img_index:
                rows.append(dict(id=iid, x=xy[0], y=xy[1], miss=gf.get(iid), fp=pf.get(iid),
                                 cluster=(cl.get(iid) if cl is not None else None)))
        _dump(out("failure_map.json"), rows)
    print(f"Recall {overview['recall']:.1%}, precision {overview['precision']:.1%} at score >= {cfg['conf_thresh']}")
    print(f"{len(cards_g)} recall findings, {len(cards_p)} precision findings -> {adir}  ({time.time() - t0:.1f}s)")
    return overview, cards_g, cards_p


def _np(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _dump(path, obj):
    def clean(x):
        if isinstance(x, float) and not math.isfinite(x):
            return None
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        return x
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(clean(obj), fh, default=_np)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--conf", type=float, default=CFG["conf_thresh"])
    ap.add_argument("--iou", type=float, default=CFG["iou_fg"])
    a = ap.parse_args()
    run(a.dir, conf_thresh=a.conf, iou_fg=a.iou)
