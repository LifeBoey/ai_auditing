"""
Generic detection metrics for any subset of images.

Each image is matched once (COCO rules: per class, greedy by score, best IoU, crowd GT absorbs
duplicates, maxDets=100, 10 IoU thresholds). A group's metrics are then computed by pooling
its images' match results, so evaluating many groups or bootstrap resamples is cheap and gives
the same mAP / AP50 as pycocotools.COCOeval restricted to those imgIds.
"""

import math

import numpy as np

IOU_THRS = np.linspace(0.5, 0.95, 10)
REC_THRS = np.linspace(0.0, 1.0, 101)
MAX_DETS = 100


def _iou(d, g, crowd):
    """xywh boxes -> (D, G) IoU; for crowd GT the denominator is the detection area (COCO)."""
    if len(d) == 0 or len(g) == 0:
        return np.zeros((len(d), len(g)))
    dx2, dy2 = d[:, 0] + d[:, 2], d[:, 1] + d[:, 3]
    gx2, gy2 = g[:, 0] + g[:, 2], g[:, 1] + g[:, 3]
    iw = np.clip(np.minimum(dx2[:, None], gx2) - np.maximum(d[:, None, 0], g[:, 0]), 0, None)
    ih = np.clip(np.minimum(dy2[:, None], gy2) - np.maximum(d[:, None, 1], g[:, 1]), 0, None)
    inter = iw * ih
    da, ga = d[:, 2] * d[:, 3], g[:, 2] * g[:, 3]
    union = np.where(crowd[None, :], da[:, None], da[:, None] + ga - inter)
    return inter / np.maximum(union, 1e-12)


class ImageEval:
    """Match result of one image: per detection score, class, tp / ignored flags per IoU threshold."""
    __slots__ = ("score", "cls", "tp", "ign", "n_gt")

    def __init__(self, score, cls, tp, ign, n_gt):
        self.score, self.cls, self.tp, self.ign, self.n_gt = score, cls, tp, ign, n_gt


def evaluate_image(gts, dts, max_dets=MAX_DETS):
    """gts: COCO annotations of one image; dts: [{"category_id", "bbox" xywh, "score"}]."""
    classes = sorted({a["category_id"] for a in gts} | {d["category_id"] for d in dts})
    T = len(IOU_THRS)
    S, C, TP, IG, n_gt = [], [], [], [], {}
    for c in classes:
        g = [a for a in gts if a["category_id"] == c]
        g.sort(key=lambda a: bool(a.get("iscrowd", 0)))                # crowd last, as COCO
        crowd = np.array([bool(a.get("iscrowd", 0)) for a in g], bool)
        n_gt[c] = int((~crowd).sum())
        d = sorted((x for x in dts if x["category_id"] == c), key=lambda x: -x["score"])[:max_dets]
        gb = np.array([a["bbox"] for a in g], float).reshape(-1, 4)
        db = np.array([x["bbox"] for x in d], float).reshape(-1, 4)
        iou = _iou(db, gb, crowd)
        tp, ign = np.zeros((T, len(d)), bool), np.zeros((T, len(d)), bool)
        for t, thr in enumerate(np.minimum(IOU_THRS, 1 - 1e-10)):
            used = np.zeros(len(g), bool)
            for i in range(len(d)):
                ok = iou[i] >= thr
                real = ok & ~crowd & ~used
                if real.any():
                    j = int(np.where(real, iou[i], -1).argmax())
                    used[j], tp[t, i] = True, True
                elif (ok & crowd).any():
                    ign[t, i] = True                                    # on a crowd region: not counted
        S.append([x["score"] for x in d])
        C.append([c] * len(d))
        TP.append(tp)
        IG.append(ign)
    return ImageEval(np.array(sum(S, []), float), np.array(sum(C, []), int),
                     np.concatenate(TP, 1) if TP else np.zeros((T, 0), bool),
                     np.concatenate(IG, 1) if IG else np.zeros((T, 0), bool), n_gt)


def pool(evals):
    """Concatenate a list of ImageEval (duplicates allowed, for bootstrapping)."""
    n_gt = {}
    for e in evals:
        for c, n in e.n_gt.items():
            n_gt[c] = n_gt.get(c, 0) + n
    T = len(IOU_THRS)
    return ImageEval(np.concatenate([e.score for e in evals]) if evals else np.zeros(0),
                     np.concatenate([e.cls for e in evals]) if evals else np.zeros(0, int),
                     np.concatenate([e.tp for e in evals], 1) if evals else np.zeros((T, 0), bool),
                     np.concatenate([e.ign for e in evals], 1) if evals else np.zeros((T, 0), bool), n_gt)


def ap_per_class(p, classes, ts=None):
    """(T, K) AP, NaN where a class has no GT in the pool (COCO excludes those).
    ts restricts the IoU thresholds computed (others stay NaN)."""
    ts = range(len(IOU_THRS)) if ts is None else ts
    out = np.full((len(IOU_THRS), len(classes)), np.nan)
    for k, c in enumerate(classes):
        npig = p.n_gt.get(c, 0)
        if npig == 0:
            continue
        m = p.cls == c
        order = np.argsort(-p.score[m], kind="mergesort")
        tp, ig = p.tp[:, m][:, order], p.ign[:, m][:, order]
        tps = np.cumsum(tp & ~ig, 1)
        fps = np.cumsum(~tp & ~ig, 1)
        for t in ts:
            rc = tps[t] / npig
            pr = tps[t] / np.maximum(tps[t] + fps[t], np.spacing(1))
            pr = np.maximum.accumulate(pr[::-1])[::-1] if len(pr) else pr
            idx = np.searchsorted(rc, REC_THRS, side="left")
            q = np.zeros(len(REC_THRS))
            ok = idx < len(pr)
            q[ok] = pr[idx[ok]]
            out[t, k] = q.mean()
    return out


def summary(p, classes):
    """mAP@[.5:.95], AP50 and per-class AP50 of a pool."""
    ap = ap_per_class(p, classes)
    with np.errstate(invalid="ignore"):
        return dict(mAP=float(np.nanmean(ap)) if np.isfinite(ap).any() else np.nan,
                    AP50=float(np.nanmean(ap[0])) if np.isfinite(ap[0]).any() else np.nan,
                    per_class_AP50={c: float(ap[0, k]) for k, c in enumerate(classes)})


def ap50(p, classes):
    a = ap_per_class(p, classes, ts=[0])[0]
    return float(np.nanmean(a)) if np.isfinite(a).any() else np.nan


def counts_at(p, conf, t=0):
    """TP / FP / FN at IoU_THRS[t] keeping detections with score >= conf (greedy-by-score matching
    makes this equal to matching only the kept detections)."""
    keep = (p.score >= conf) & ~p.ign[t]
    tp = int((p.tp[t] & keep).sum())
    return tp, int(keep.sum()) - tp, sum(p.n_gt.values()) - tp


def best_f1_conf(p, t=0):
    """Score threshold with the highest F1 at IoU_THRS[t]."""
    keep = ~p.ign[t]
    s, tp = p.score[keep], p.tp[t][keep]
    order = np.argsort(-s, kind="mergesort")
    s, tp = s[order], tp[order]
    n_gt = sum(p.n_gt.values())
    ctp = np.cumsum(tp)
    f1 = 2 * ctp / (np.arange(1, len(s) + 1) + n_gt)
    last = np.r_[s[1:] != s[:-1], True]                 # evaluate only after the last of tied scores
    i = int(np.argmax(np.where(last, f1, -1)))
    return float(s[i]), float(f1[i])


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else np.nan
    r = tp / (tp + fn) if tp + fn else np.nan
    f = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan
    return p, r, f


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def two_prop_p(k1, n1, k2, n2):
    """Two-sided p-value that k1/n1 and k2/n2 come from the same rate."""
    if min(n1, n2) == 0:
        return np.nan
    pp = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    return math.erfc(abs(k1 / n1 - k2 / n2) / se / math.sqrt(2))


def bh(p):
    """Benjamini-Hochberg adjusted p-values (NaN passed through)."""
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if len(ok) == 0:
        return out
    q = p[ok]
    order = np.argsort(q)
    adj = np.empty(len(q))
    adj[order] = np.minimum.accumulate((q[order] * len(q) / np.arange(1, len(q) + 1))[::-1])[::-1]
    out[ok] = np.minimum(adj, 1.0)
    return out
