"""Aspects that need only the pixels."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

from .base import Aspect


def _stats(path):
    im = Image.open(path)
    W, H = im.size
    rgb = im.convert("RGB")
    g = np.asarray(rgb.convert("L"), np.float32)
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    hsv = np.asarray(rgb.convert("HSV"), np.float32)
    h, s = hsv[..., 0] * (2 * np.pi / 255), hsv[..., 1] / 255
    # saturation-weighted circular mean: grey pixels carry no hue
    hue = np.degrees(np.arctan2((s * np.sin(h)).sum(), (s * np.cos(h)).sum())) % 360 if s.sum() > 0 else np.nan
    return dict(brightness=g.mean(), contrast=g.std(), sharpness=lap.var(), saturation=s.mean(),
                hue=hue, aspect_ratio=W / H, megapixels=W * H / 1e6)


def pixel_stats(ctx):
    def run():
        with ThreadPoolExecutor(8) as pool:
            rows = list(pool.map(_stats, [ctx.path(f) for f in ctx.files]))
        return pd.DataFrame(rows, index=pd.Index(ctx.files, name="file_name"))
    return ctx.cached("pixel_stats", run)


def _make(stat, desc):
    class _A(Aspect):
        name = stat
        kind = "numeric"
        requires = ("images",)
        description = desc

        def compute(self, ctx):
            v = pixel_stats(ctx)[stat]
            if stat == "hue":
                # hue is circular: unwrap to within +-180 deg of the dataset's mean hue so that
                # 359 and 1 land next to each other instead of at opposite ends of the bins
                r = np.radians(v.dropna())
                mean = np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean()))
                v = (v - mean + 180) % 360 - 180 + mean
            return v
    _A.__name__ = f"{stat.title()}Aspect"
    return _A


PIXEL_ASPECTS = [
    _make("brightness", "mean grey level (0-255)"),
    _make("contrast", "std of grey level"),
    _make("sharpness", "variance of the Laplacian (low = blurry)"),
    _make("saturation", "mean HSV saturation (0-1)"),
    _make("hue", "saturation-weighted mean hue (degrees, unwrapped around the dataset mean) - stain colour"),
    _make("aspect_ratio", "width / height"),
    _make("megapixels", "image resolution"),
]
