"""
Load someone else's aspect from a .py file (see plugin_template.py for the contract).

The file is imported by path, so it can live anywhere and needs no import from cv_aspects.
It must define compute(image_paths, labels=None, options=None) and may define NAME, KIND,
DESCRIPTION and NEEDS_LABELS.
"""

import importlib.util
import os

import numpy as np
import pandas as pd

from .base import Aspect

KINDS = ("categorical", "numeric", "embedding")


def kmeans_groups(emb, k, seed=0):
    """Cluster vectors; clusters are named by size (c0 = largest) so names are stable between runs."""
    from sklearn.cluster import KMeans

    emb = np.asarray(emb, float)
    k = min(int(k), len(emb))
    lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(emb)
    order = {c: i for i, c in enumerate(pd.Series(lab).value_counts().index)}
    return [f"c{order[c]}" for c in lab]


def _labels_for_plugin(ctx):
    """{file_name: [{"bbox": [x, y, w, h], "category": name, "iscrowd": 0/1}]} for the plugin."""
    if ctx.gt is None:
        return None
    return {f: [dict(bbox=list(a["bbox"]), category=ctx.categories[a["category_id"]],
                     iscrowd=int(a.get("iscrowd", 0))) for a in ctx.anns_by_file.get(f, [])]
            for f in ctx.files}


def load_plugin(path):
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise SystemExit(f"plugin not found: {path}")
    stem = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(f"cv_aspects_plugin_{stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not callable(getattr(mod, "compute", None)):
        raise SystemExit(f"plugin {path} must define compute(image_paths, labels=None, options=None)")
    kind = getattr(mod, "KIND", "categorical")
    if kind not in KINDS:
        raise SystemExit(f"plugin {path}: KIND must be one of {KINDS}, got {kind!r}")
    return PluginAspect(mod, path, getattr(mod, "NAME", stem), kind)


class PluginAspect(Aspect):
    def __init__(self, mod, path, name, kind):
        self.mod, self.path, self.name = mod, path, name
        self.source_kind = kind
        self.kind = "categorical" if kind == "embedding" else kind      # clusters are categories
        self.requires = ("images", "labels") if getattr(mod, "NEEDS_LABELS", False) else ("images",)
        self.description = getattr(mod, "DESCRIPTION", "") + f"  [plugin: {path}]"

    def compute(self, ctx):
        options = dict(ctx.config.get(self.name, {}))
        paths = [ctx.path(f) for f in ctx.files]
        out = self.mod.compute(paths, labels=_labels_for_plugin(ctx), options=options)

        if isinstance(out, dict):                                       # keyed by path or file name
            by_name = {os.path.basename(k): v for k, v in out.items()}
            out = [by_name.get(f) for f in ctx.files]
        if self.source_kind == "embedding":
            emb = np.asarray(out, float)
            if emb.ndim != 2 or len(emb) != len(ctx.files):
                raise SystemExit(f"plugin {self.name}: KIND='embedding' must return one vector per image "
                                 f"(shape ({len(ctx.files)}, D)), got shape {emb.shape}")
            return ctx.series(kmeans_groups(emb, options.get("k", 5)))
        out = list(out)
        if len(out) != len(ctx.files):
            raise SystemExit(f"plugin {self.name}: compute() returned {len(out)} values for "
                             f"{len(ctx.files)} images - return one value per image, in the same order")
        s = pd.Series(out, index=pd.Index(ctx.files, name="file_name"), dtype=object)
        return pd.to_numeric(s, errors="coerce") if self.kind == "numeric" else s
