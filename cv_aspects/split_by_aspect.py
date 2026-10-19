"""
split_by_aspect.py -- stage 1: put every image into a group, for each aspect.

Inputs: images (required), GT json (optional, enables label aspects), CLIP weights (optional,
enables clip_cluster). Output: <out>/tags.csv with one row per image and, per aspect,
<aspect> (raw value) and <aspect>__group (group label); <out>/tags.json with how each aspect was
binned. Stage 2 (evaluate_by_aspect.py) only reads these files, so a hand-written tags.csv with
file_name + any *__group columns works too.

    python split_by_aspect.py --images ../DATA/bccd_images --labels ../DATA/bccd_ground_truth.json \
        --clip C:/Users/Gabriel/fiftyone/__models__/CLIP-ViT-B-32.pt
    python split_by_aspect.py --images ... --aspects brightness,n_gt --config aspects_config.example.json
    python split_by_aspect.py --images ... --plugin my_aspect.py --aspects plugins   # your own aspect
    python split_by_aspect.py --list [--plugin my_aspect.py]                         # show known aspects

Your own aspect: copy plugin_template.py, fill in the marked parts, pass it with --plugin
(or list it under "plugins" in the --config JSON). See plugins/odd_even.py for a worked example.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from aspects import Context, available, load_plugin
from binning import bin_categorical, bin_numeric

HERE = os.path.dirname(os.path.abspath(__file__))


def split(ctx, names, plugins=(), min_groups=2):
    known = available(ctx, plugins)
    unknown = [n for n in names if n not in known]
    if unknown:
        raise SystemExit(f"unknown aspects {unknown}; known: {sorted(known)}")
    default_bins = int(ctx.config.get("default_n_bins", 4))
    tags = pd.DataFrame({"image_id": [ctx.image_id(f) for f in ctx.files]},
                        index=pd.Index(ctx.files, name="file_name"))
    meta, skipped, dropped = {}, {}, {}
    for n in names:
        a = known[n]
        miss = [r for r in a.requires if not ctx.has(r)]
        if miss:
            skipped[n] = f"needs {', '.join('--' + r for r in miss)}"
            continue
        raw = a.compute(ctx).reindex(tags.index)
        cfg = ctx.config.get(n, {})
        groups, info = (bin_numeric(raw, cfg, default_bins) if a.kind == "numeric" else bin_categorical(raw))
        counts = groups.value_counts()
        if (counts > 0).sum() < min_groups:
            dropped[n] = f"only one group ({counts.index[0]}) - the aspect is constant on this data"
            continue
        tags[n] = raw.round(4) if a.kind == "numeric" else raw
        tags[f"{n}__group"] = groups
        meta[n] = dict(kind=a.kind, description=a.description, requires=list(a.requires), **info,
                       counts={g: int(counts.get(g, 0)) for g in info["groups"]})
        if hasattr(a, "path"):
            meta[n].update(plugin=a.path, plugin_kind=a.source_kind)
    return tags, meta, skipped, dropped


def montage(ctx, tags, meta, out_dir, per_group=6, thumb=96):
    """One thumbnail strip per group, to see what a group actually looks like."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    for n, m in meta.items():
        gs = [g for g in m["groups"] if m["counts"][g]]
        fig, axs = plt.subplots(len(gs), per_group, figsize=(per_group * 1.3, len(gs) * 1.15), squeeze=False)
        for r, g in enumerate(gs):
            files = tags.index[tags[f"{n}__group"] == g].tolist()
            pick = rng.choice(files, min(per_group, len(files)), replace=False)
            for c in range(per_group):
                ax = axs[r, c]
                ax.axis("off")
                if c < len(pick):
                    im = Image.open(ctx.path(pick[c])).convert("RGB")
                    im.thumbnail((thumb, thumb))
                    ax.imshow(im)
            axs[r, 0].text(-0.08, 0.5, f"{g}\n(n={m['counts'][g]})", transform=axs[r, 0].transAxes,
                           ha="right", va="center", fontsize=7)
        fig.suptitle(n, fontsize=9)
        fig.subplots_adjust(left=0.22, wspace=0.05, hspace=0.1)
        fig.savefig(os.path.join(out_dir, f"groups_{n}.png"), dpi=110)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=False)
    ap.add_argument("--labels", help="COCO GT json (enables label-derived aspects)")
    ap.add_argument("--clip", help="CLIP weights, e.g. __models__/CLIP-ViT-B-32.pt (enables clip_cluster)")
    ap.add_argument("--plugin", action="append", default=[], metavar="FILE.py",
                    help="your own aspect (see plugin_template.py); repeat for several")
    ap.add_argument("--aspects", default="all",
                    help="comma-separated names, 'all' (built-ins + plugins) or 'plugins' (plugins only)")
    ap.add_argument("--config", help="JSON with per-aspect binning (see aspects_config.example.json)")
    ap.add_argument("--out", default=os.path.join(HERE, "aspects_out"))
    ap.add_argument("--no-montage", action="store_true", help="skip the per-group thumbnail images")
    ap.add_argument("--list", action="store_true", help="list known aspects and exit")
    args = ap.parse_args()

    config = json.load(open(args.config, encoding="utf-8")) if args.config else {}
    # plugin paths from the config are relative to the config file
    cfg_dir = os.path.dirname(os.path.abspath(args.config)) if args.config else os.getcwd()
    plugin_files = args.plugin + [os.path.join(cfg_dir, p) for p in config.get("plugins", [])]
    plugins = [load_plugin(p) for p in plugin_files]
    if args.list:
        class _Dummy:
            gt = None
        for n, a in available(_Dummy(), plugins).items():
            kind = getattr(a, "source_kind", a.kind)
            print(f"  {n:<16} {kind:<12} needs {'/'.join(a.requires):<13} {a.description}")
        print("  n_<class>, has_<class> are added per class when --labels is given")
        return
    if not args.images:
        ap.error("--images is required")

    os.makedirs(args.out, exist_ok=True)
    ctx = Context(args.images, args.labels, args.clip, cache_dir=args.out, config=config)
    if args.aspects == "all":
        names = config.get("aspects") or list(available(ctx, plugins))
    elif args.aspects == "plugins":
        names = [p.name for p in plugins]
    else:
        names = [s.strip() for s in args.aspects.split(",") if s.strip()]
    print(f"{len(ctx.files)} images, {len(names)} aspects requested")

    tags, meta, skipped, dropped = split(ctx, names, plugins)
    tags.reset_index().to_csv(os.path.join(args.out, "tags.csv"), index=False)
    json.dump(dict(images=os.path.abspath(args.images), labels=args.labels and os.path.abspath(args.labels),
                   clip=args.clip, aspects=meta, skipped=skipped, dropped=dropped),
              open(os.path.join(args.out, "tags.json"), "w", encoding="utf-8"), indent=2)

    for n, m in meta.items():
        print(f"  {n:<16} {m['method']:<12} " + "  ".join(f"{g}:{c}" for g, c in m["counts"].items()))
    for n, why in skipped.items():
        print(f"  [skip] {n}: {why}")
    for n, why in dropped.items():
        print(f"  [drop] {n}: {why}")
    if not args.no_montage and meta:
        montage(ctx, tags, meta, os.path.join(args.out, "plots"))
    print(f"wrote {os.path.join(args.out, 'tags.csv')} and tags.json")


if __name__ == "__main__":
    main()
