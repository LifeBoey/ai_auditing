"""Unsupervised visual groups: KMeans on L2-normalised CLIP image embeddings (needs --clip)."""

import os
import time

import numpy as np

from .base import Aspect
from .plugin import kmeans_groups


def clip_embeddings(ctx, batch=64):
    def run():
        cache = os.path.join(ctx.cache_dir, "clip_emb.npz") if ctx.cache_dir else None
        if cache and os.path.isfile(cache):
            z = np.load(cache, allow_pickle=False)
            if list(z["files"]) == ctx.files and str(z["weights"]) == os.path.abspath(ctx.clip_weights):
                print(f"CLIP: reusing {cache}")
                return z["emb"]
        import open_clip
        import torch
        from PIL import Image

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # fp32: the OpenAI JIT checkpoint loads as fp16 on GPU otherwise
        model = open_clip.load_openai_model(ctx.clip_weights, precision="fp32", device=device).eval()
        prep = open_clip.image_transform(model.visual.image_size, is_train=False)
        out, t0 = [], time.time()
        for s in range(0, len(ctx.files), batch):
            x = torch.stack([prep(Image.open(ctx.path(f)).convert("RGB")) for f in ctx.files[s:s + batch]])
            with torch.no_grad():
                e = model.encode_image(x.to(device)).float()
            out.append((e / e.norm(dim=-1, keepdim=True)).cpu().numpy())
            print(f"\r  CLIP {min(s + batch, len(ctx.files))}/{len(ctx.files)} ({time.time() - t0:.0f}s)", end="")
        print()
        emb = np.concatenate(out)
        if cache:
            np.savez(cache, emb=emb, files=np.array(ctx.files), weights=os.path.abspath(ctx.clip_weights))
        return emb
    return ctx.cached("clip_emb", run)


class ClipClusterAspect(Aspect):
    name = "clip_cluster"
    kind = "categorical"
    requires = ("clip",)
    description = "KMeans cluster of the CLIP image embedding"

    def compute(self, ctx):
        k = int(ctx.config.get(self.name, {}).get("k", 5))
        return ctx.series(kmeans_groups(clip_embeddings(ctx), k))
