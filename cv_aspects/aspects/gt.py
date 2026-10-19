"""Aspects derived from the ground-truth boxes (need --labels). Crowd annotations are ignored."""

import numpy as np
import pandas as pd

from .base import Aspect

SMALL = 32 ** 2   # COCO "small" box area, px^2


def _boxes(anns):
    b = np.array([a["bbox"] for a in anns if not a.get("iscrowd", 0)], float).reshape(-1, 4)
    return b[(b[:, 2] > 0) & (b[:, 3] > 0)]


def _crowding(b):
    """Mean over boxes of the max IoU with any other GT box (0 when fewer than 2 boxes)."""
    if len(b) < 2:
        return 0.0
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(x2[:, None], x2) - np.maximum(x1[:, None], x1), 0, None)
    ih = np.clip(np.minimum(y2[:, None], y2) - np.maximum(y1[:, None], y1), 0, None)
    inter = iw * ih
    area = b[:, 2] * b[:, 3]
    iou = inter / (area[:, None] + area - inter)
    np.fill_diagonal(iou, 0)
    return float(iou.max(1).mean())


def gt_stats(ctx):
    def run():
        rows = []
        for f in ctx.files:
            im = ctx.gt_images.get(f)
            if im is None:
                rows.append({})
                continue
            anns = [a for a in ctx.anns_by_file[f] if not a.get("iscrowd", 0)]
            b = _boxes(anns)
            area = b[:, 2] * b[:, 3]
            cls = [ctx.categories[a["category_id"]] for a in anns]
            r = dict(n_gt=len(anns), n_classes=len(set(cls)), crowding=_crowding(b),
                     small_frac=float((area < SMALL).mean()) if len(b) else np.nan,
                     mean_box_area=float(area.mean() / (im["width"] * im["height"])) if len(b) else np.nan)
            for c in ctx.categories.values():
                r[f"n_{c}"] = cls.count(c)
            rows.append(r)
        return pd.DataFrame(rows, index=pd.Index(ctx.files, name="file_name"))
    return ctx.cached("gt_stats", run)


def _make(col, kind, desc, fn=None):
    class _A(Aspect):
        name = col if fn is None else fn[0]
        requires = ("labels",)
        description = desc

        def compute(self, ctx):
            s = gt_stats(ctx)[col]
            return s if fn is None else fn[1](s)
    _A.kind = kind
    return _A


def gt_aspects(ctx):
    """Fixed aspects plus one count and one has/has-not aspect per class."""
    out = [_make("n_gt", "numeric", "number of GT boxes"),
           _make("n_classes", "categorical", "number of distinct classes present"),
           _make("crowding", "numeric", "mean max IoU between GT boxes"),
           _make("small_frac", "numeric", f"fraction of GT boxes smaller than {int(SMALL ** .5)}^2 px"),
           _make("mean_box_area", "numeric", "mean GT box area / image area")]
    if ctx.gt is not None:
        for c in ctx.categories.values():
            out.append(_make(f"n_{c}", "numeric", f"number of {c} boxes"))
            out.append(_make(f"n_{c}", "categorical", f"image contains a {c}",
                             fn=(f"has_{c}", lambda s: s.map(lambda v: "yes" if v > 0 else "no"
                                                             if v == v else np.nan))))
    return out
