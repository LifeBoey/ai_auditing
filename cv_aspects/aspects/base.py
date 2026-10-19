"""
The black-box contract. An aspect maps every image to one value:

    class MyAspect(Aspect):
        name = "my_aspect"
        kind = "numeric"            # numeric -> binned into groups later; categorical -> used as-is
        requires = ("images",)      # any of: images, labels, clip
        def compute(self, ctx):     # -> pd.Series indexed by file_name
            ...

Expensive work shared by several aspects (reading pixels, embedding with CLIP) is cached on the
Context so each image is read / embedded once per run.
"""

import json
import os

import numpy as np
import pandas as pd

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


class Aspect:
    name = ""
    kind = "numeric"
    requires = ("images",)
    description = ""

    def compute(self, ctx):
        raise NotImplementedError


class Context:
    def __init__(self, images_dir, labels_path=None, clip_weights=None, cache_dir=None, config=None):
        self.images_dir = images_dir
        self.files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith(IMAGE_EXTS))
        if not self.files:
            raise SystemExit(f"no images found in {images_dir}")
        self.clip_weights = clip_weights
        self.cache_dir = cache_dir
        self.config = config or {}
        self._cache = {}

        self.gt = None
        if labels_path:
            self.gt = json.load(open(labels_path, encoding="utf-8"))
            by_file = {im["file_name"]: im for im in self.gt["images"]}
            missing = [f for f in self.files if f not in by_file]
            if missing:
                print(f"[warn] {len(missing)} images are not in {labels_path} (e.g. {missing[0]}); "
                      f"label-based aspects will be empty for them")
            self.gt_images = by_file
            self.categories = {c["id"]: c["name"] for c in self.gt["categories"]}
            anns = {}
            for a in self.gt["annotations"]:
                anns.setdefault(a["image_id"], []).append(a)
            self.anns_by_file = {f: anns.get(im["id"], []) for f, im in by_file.items()}

    def has(self, req):
        return {"images": True, "labels": self.gt is not None, "clip": bool(self.clip_weights)}[req]

    def path(self, f):
        return os.path.join(self.images_dir, f)

    def image_id(self, f):
        im = self.gt_images.get(f) if self.gt is not None else None
        return im["id"] if im else None

    def cached(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def series(self, values):
        """Helper: list aligned with self.files -> Series indexed by file_name."""
        return pd.Series(np.asarray(values), index=pd.Index(self.files, name="file_name"))
