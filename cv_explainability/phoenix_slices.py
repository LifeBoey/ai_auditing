"""
phoenix_slices.py -- where does the detector perform poorly? Slice correct/incorrect in Arize Phoenix.

1. Match predictions.json to labels.json (COCO-style greedy matching, one IoU threshold), so every
   ground-truth object and every prediction is marked correct or incorrect, with the reason:
     ground truth  detected | missed | confused (right place, wrong class) | mislocalized
                   | claimed_by_neighbour (a matching box exists but went to an adjacent object)
     prediction    tp | background | wrong_class | mislocalized | duplicate
2. Attach slice attributes to each one: category, supercategory, box area / size bucket, aspect,
   truncation, overlap with other objects, position in frame, objects per image, image size, and
   with --image-tags, CLIP scene tags (indoor/outdoor, setting, lighting, ...) plus brightness,
   contrast and sharpness.
3. Print and save a slice table: failure rate per attribute value vs the rest of the data, with
   lift, Wilson 95% CI and a Benjamini-Hochberg-adjusted one-sided p-value.
4. Send everything to Phoenix: one trace per image, one span per object/prediction, attributes in
   the span metadata, and `correct` / `outcome` annotations. In the Phoenix UI, filter on metadata
   and the annotation summary gives that slice's accuracy.

Phoenix 20.x has no inferences/embeddings API any more, so this uses plain OpenTelemetry plus
Phoenix's REST API, and never imports `phoenix` (the client only needs a running server).

Run (start the Phoenix server first:  phoenix serve   ->  http://localhost:6006)
    python phoenix_slices.py                      # defaults: ../predictions.json, ../labels.json
    python phoenix_slices.py --image-tags         # + CLIP tags from __models__/CLIP-ViT-B-32.pt
    python phoenix_slices.py --no-phoenix         # analysis + CSVs only
    python phoenix_slices.py --replace            # delete and rebuild the Phoenix project

Phoenix span filters to try (Spans tab, filter box):
    name == 'gt_object' and metadata['size'] == 'small'
    name == 'gt_object' and metadata['category'] == 'person' and annotations['correct'].label == 'incorrect'
    name == 'prediction' and annotations['outcome'].label == 'background' and metadata['score'] > 0.9
    name == 'image' and annotations['recall'].score < 0.5
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
VAL_DIR = os.path.dirname(HERE)
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(VAL_DIR)), "__models__")

IOU = 0.5          # IoU to count an object as found
IOU_BG = 0.1       # below this a prediction overlaps nothing
SIZE_EDGES = (32 ** 2, 96 ** 2)   # COCO small / medium / large (box area, px^2)

GT_CAT = ["category", "supercategory", "size", "aspect", "truncated", "occlusion", "position",
          "n_objects_bucket", "img_orientation"]
GT_NUM = ["area_px", "aspect_ratio", "overlap", "n_objects", "img_megapixels"]
PRED_CAT = ["category", "supercategory", "size", "aspect", "truncated", "position", "score_bin",
            "n_objects_bucket", "img_orientation"]
PRED_NUM = ["score", "area_px", "aspect_ratio", "n_objects", "img_megapixels"]
IMG_NUM_TAGS = ["brightness", "contrast", "sharpness"]

CLIP_PROMPTS = {
    "scene": {"indoor": "a photo taken indoors", "outdoor": "a photo taken outdoors"},
    "setting": {"street": "a photo of a city street", "nature": "a photo of nature or a landscape",
                "home": "a photo inside a home", "sports": "a photo of a sports event",
                "food": "a close-up photo of food", "shop/office": "a photo inside a shop or office",
                "vehicle/transport": "a photo of vehicles, trains or planes"},
    "lighting": {"well lit": "a brightly lit photo", "dark": "a dark, poorly lit photo"},
    "focus": {"sharp": "a sharp, in-focus photo", "blurry": "a blurry, out-of-focus photo"},
    "color": {"color": "a color photo", "black and white": "a black and white photo"},
}


# ============================================================
# load + match
# ============================================================
def load(preds_path, labels_path, images_dir):
    labels = json.load(open(labels_path, encoding="utf-8"))
    preds = json.load(open(preds_path, encoding="utf-8"))
    cats = {c["id"]: c for c in labels["categories"]}
    by_file = {im["file_name"]: im for im in labels["images"]}
    anns = {}
    for a in labels["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    missing = [r["file_name"] for r in preds if r["file_name"] not in by_file]
    if missing:
        print(f"[warn] {len(missing)} predicted images are not in {labels_path}, skipped (e.g. {missing[0]})")
    records = []
    for r in preds:
        im = by_file.get(r["file_name"])
        if im is not None:
            records.append((r, im, anns.get(im["id"], [])))
    print(f"Loaded {len(records)} images ({len(by_file) - len(records)} labelled images have no "
          f"predictions entry and are left out), {len(cats)} categories")
    return records, cats


def iou_xyxy(a, b):
    """IoU (n, m) and intersection over a's area (n, m)."""
    if len(a) == 0 or len(b) == 0:
        z = np.zeros((len(a), len(b)))
        return z, z
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9), inter / np.maximum(area_a[:, None], 1e-9)


def match_image(gb, gc, crowd, pb, pc, ps, thr=IOU):
    """Greedy by score, same class, IoU >= thr; preds on crowd regions are ignored (COCO rules)."""
    ng, npr = len(gb), len(pb)
    iou, ioa = iou_xyxy(pb, gb)
    ov = np.where(crowd[None, :], 0.0, iou)          # overlap with real (non-crowd) objects
    same = pc[:, None] == gc[None, :]
    taken, g_match = np.zeros(ng, bool), np.full(ng, -1)
    p_out, p_iou, p_gt = [""] * npr, np.zeros(npr), np.full(npr, -1)

    for i in np.argsort(-ps, kind="stable"):
        if ng:
            cand = np.where(same[i] & ~crowd & ~taken, ov[i], -1.0)
            j = int(cand.argmax())
            if cand[j] >= thr:
                taken[j], g_match[j] = True, i
                p_out[i], p_iou[i], p_gt[i] = "tp", cand[j], j
                continue
            if (same[i] & crowd & (ioa[i] >= thr)).any():
                p_out[i] = "ignored"
                continue
    for i in range(npr):
        if p_out[i]:
            continue
        best = ov[i].max() if ng else 0.0
        same_best = np.where(same[i], ov[i], 0).max() if ng else 0.0
        other_best = np.where(~same[i], ov[i], 0).max() if ng else 0.0
        p_out[i] = ("duplicate" if same_best >= thr else "wrong_class" if other_best >= thr
                    else "mislocalized" if best >= IOU_BG else "background")
        p_iou[i] = best
        p_gt[i] = int(ov[i].argmax()) if best > 0 else -1

    valid = np.array([o != "ignored" for o in p_out], bool)
    g_out, g_iou, g_pred = [""] * ng, np.zeros(ng), np.full(ng, -1)
    for j in range(ng):
        if crowd[j]:
            g_out[j] = "crowd"
            continue
        if taken[j]:
            g_out[j], g_iou[j], g_pred[j] = "detected", ov[g_match[j], j], g_match[j]
            continue
        col = np.where(valid, ov[:, j], 0) if npr else np.zeros(0)
        if not npr or col.max() <= 0:
            g_out[j] = "missed"
            continue
        same_col = np.where(same[:, j], col, 0)
        other_col = np.where(~same[:, j], col, 0)
        if same_col.max() >= thr:
            g_out[j], k = "claimed_by_neighbour", int(same_col.argmax())
        elif other_col.max() >= thr:
            g_out[j], k = "confused", int(other_col.argmax())
        elif col.max() >= IOU_BG:
            g_out[j], k = "mislocalized", int(col.argmax())
        else:
            g_out[j], k = "missed", int(col.argmax())
        g_iou[j], g_pred[j] = col[k], k
    return g_out, g_iou, g_pred, p_out, p_iou, p_gt


# ============================================================
# units + attributes
# ============================================================
def _bucket(v, edges, names):
    for e, n in zip(edges, names):
        if v < e:
            return n
    return names[-1]


def _box_attrs(x1, y1, x2, y2, W, H):
    w, h = max(x2 - x1, 0.0), max(y2 - y1, 0.0)
    area = w * h
    ar = w / h if h > 0 else 0.0
    cx, cy = (x1 + x2) / 2 / W - 0.5, (y1 + y2) / 2 / H - 0.5
    cd = math.hypot(cx, cy) / math.hypot(0.5, 0.5)
    return dict(area_px=round(area, 1), rel_area=round(area / (W * H), 5),
                size=_bucket(area, SIZE_EDGES, ["small", "medium", "large"]),
                aspect_ratio=round(ar, 3), aspect=_bucket(ar, (0.5, 2.0), ["tall", "squarish", "wide"]),
                truncated=bool(x1 <= 1 or y1 <= 1 or x2 >= W - 1 or y2 >= H - 1),
                center_dist=round(cd, 3), position=_bucket(cd, (0.33, 0.66), ["center", "middle", "edge"]))


def _img_attrs(im, n_obj, n_cls, n_pred):
    W, H = im["width"], im["height"]
    return dict(img_width=W, img_height=H, img_megapixels=round(W * H / 1e6, 3),
                img_orientation="landscape" if W > H * 1.1 else "portrait" if H > W * 1.1 else "square",
                n_objects=n_obj, n_objects_bucket=_bucket(n_obj, (1, 2, 4, 8, 16), ["0", "1", "2-3", "4-7", "8-15", "16+"]),
                n_classes=n_cls, n_preds=n_pred)


def build_units(records, cats, images_dir):
    gt_rows, pred_rows, img_rows = [], [], []
    for r, im, anns in records:
        iid, W, H = os.path.splitext(r["file_name"])[0], im["width"], im["height"]
        anns = [a for a in anns if a["bbox"][2] > 0 and a["bbox"][3] > 0]
        gb = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                       for a in anns], float).reshape(-1, 4)
        gc = np.array([a["category_id"] for a in anns], int)
        crowd = np.array([bool(a.get("iscrowd", 0)) for a in anns], bool)
        P = r["prediction"]
        pb = np.array([d["bbox"] for d in P], float).reshape(-1, 4)       # x1, y1, x2, y2 (pixels)
        pc = np.array([d["class_id"] for d in P], int)
        ps = np.array([d["score"] for d in P], float)
        g_out, g_iou, g_pred, p_out, p_iou, p_gt = match_image(gb, gc, crowd, pb, pc, ps, IOU)

        real = ~crowd
        gt_ov, _ = iou_xyxy(gb[real], gb[real])
        np.fill_diagonal(gt_ov, 0)
        overlap = dict(zip(np.flatnonzero(real), gt_ov.max(1) if len(gt_ov) else []))
        n_obj = int(real.sum())
        n_pred = sum(o != "ignored" for o in p_out)
        img = dict(image_id=iid, coco_id=im["id"], file_name=r["file_name"],
                   filepath=os.path.join(images_dir, r["file_name"]),
                   **_img_attrs(im, n_obj, len(set(gc[real].tolist())), n_pred))
        name = lambda c: cats[int(c)]["name"]

        for j, a in enumerate(anns):
            if crowd[j]:
                continue
            k, o = int(g_pred[j]), g_out[j]
            pl, sc = (name(pc[k]), float(ps[k])) if k >= 0 else (None, None)
            explain = {"detected": f"found as {pl} (IoU {g_iou[j]:.2f}, score {sc and round(sc, 2)})",
                       "confused": f"box in the right place but called {pl} (IoU {g_iou[j]:.2f}, score {sc and round(sc, 2)})",
                       "mislocalized": f"closest box ({pl}) only reaches IoU {g_iou[j]:.2f} < {IOU}",
                       "claimed_by_neighbour": f"a {pl} box covers it (IoU {g_iou[j]:.2f}) but was matched to a neighbouring object",
                       "missed": "no prediction overlaps it" if k < 0 else f"nothing close: best overlap IoU {g_iou[j]:.2f} ({pl})"}[o]
            gt_rows.append(dict(image_id=iid, gt_idx=j, ann_id=a["id"], category=name(gc[j]),
                                supercategory=cats[int(gc[j])]["supercategory"],
                                x1=gb[j, 0], y1=gb[j, 1], x2=gb[j, 2], y2=gb[j, 3],
                                **_box_attrs(*gb[j], W, H), overlap=round(float(overlap.get(j, 0.0)), 3),
                                outcome=o, correct=o == "detected", iou=round(float(g_iou[j]), 3),
                                pred_label=pl, pred_score=sc, explain=explain, **_img_extra(img)))
        for i, d in enumerate(P):
            o = p_out[i]
            if o == "ignored":
                continue
            k = int(p_gt[i])
            gl = name(gc[k]) if k >= 0 else None
            explain = {"tp": f"matches a {gl} (IoU {p_iou[i]:.2f})",
                       "duplicate": f"second box on a {gl} that is already detected (IoU {p_iou[i]:.2f})",
                       "wrong_class": f"sits on a {gl} (IoU {p_iou[i]:.2f}) but says {d['class_name']}",
                       "mislocalized": f"overlaps a {gl} only at IoU {p_iou[i]:.2f} < {IOU}",
                       "background": "nothing annotated under this box"}[o]
            pred_rows.append(dict(image_id=iid, pred_idx=i, category=name(pc[i]),
                                  supercategory=cats[int(pc[i])]["supercategory"],
                                  x1=pb[i, 0], y1=pb[i, 1], x2=pb[i, 2], y2=pb[i, 3],
                                  score=round(float(ps[i]), 4),
                                  score_bin=_bucket(ps[i], (0.6, 0.7, 0.8, 0.9), ["0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]),
                                  **_box_attrs(*pb[i], W, H), outcome=o, correct=o == "tp",
                                  iou=round(float(p_iou[i]), 3), gt_label=gl, explain=explain, **_img_extra(img)))
        g_ok = sum(o == "detected" for o in g_out)
        p_ok = sum(o == "tp" for o in p_out)
        img.update(n_detected=g_ok, n_tp=p_ok,
                   recall=round(g_ok / n_obj, 4) if n_obj else None,
                   precision=round(p_ok / n_pred, 4) if n_pred else None)
        img_rows.append(img)
    return pd.DataFrame(gt_rows), pd.DataFrame(pred_rows), pd.DataFrame(img_rows)


def _img_extra(img):
    """Image-level attributes copied onto every unit so a unit can be sliced by them directly."""
    keys = ["img_width", "img_height", "img_megapixels", "img_orientation", "n_objects", "n_objects_bucket",
            "n_classes", "n_preds"]
    return {k: img[k] for k in keys}


# ============================================================
# optional: CLIP scene tags + pixel statistics
# ============================================================
def image_tags(images, models_dir, out_dir, batch=64):
    cache = os.path.join(out_dir, "image_tags.csv")
    if os.path.isfile(cache):
        t = pd.read_csv(cache, dtype={"image_id": str})
        if set(images["image_id"]) <= set(t["image_id"]):
            print(f"Image tags: reusing {cache}")
            return t
    import open_clip
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image

    weights = os.path.join(models_dir, "CLIP-ViT-B-32.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # fp32: the JIT checkpoint loads as fp16 on GPU, which breaks the fp32 preprocessing
    model = open_clip.load_openai_model(weights, precision="fp32", device=device).eval()
    preprocess = open_clip.image_transform(model.visual.image_size, is_train=False)
    tok = open_clip.get_tokenizer("ViT-B-32")
    with torch.no_grad():
        text = {g: model.encode_text(tok(list(p.values())).to(device)) for g, p in CLIP_PROMPTS.items()}
        text = {g: t / t.norm(dim=-1, keepdim=True) for g, t in text.items()}

    def read(path):
        im = Image.open(path).convert("RGB")
        g = np.asarray(im.convert("L"), np.float32)
        lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
        return preprocess(im), dict(brightness=float(g.mean()), contrast=float(g.std()), sharpness=float(lap.var()))

    rows, ids, paths = [], images["image_id"].tolist(), images["filepath"].tolist()
    t0 = time.time()
    with ThreadPoolExecutor(8) as pool:
        for s in range(0, len(ids), batch):
            got = list(pool.map(read, paths[s:s + batch]))
            x = torch.stack([g[0] for g in got]).to(device)
            with torch.no_grad():
                f = model.encode_image(x)
                f = f / f.norm(dim=-1, keepdim=True)
            for g, t in text.items():
                prob = (100 * f @ t.T).softmax(-1).cpu().numpy()
                names = list(CLIP_PROMPTS[g])
                for n, (iid, p) in enumerate(zip(ids[s:s + batch], prob)):
                    if g == "scene":
                        rows.append(dict(image_id=iid, **got[n][1]))
                    rows[s + n][f"clip_{g}"] = names[int(p.argmax())]
                    rows[s + n][f"clip_{g}_conf"] = round(float(p.max()), 3)
            print(f"\r  CLIP tags {min(s + batch, len(ids))}/{len(ids)}  ({time.time() - t0:.0f}s)", end="")
    print()
    t = pd.DataFrame(rows)
    for c in IMG_NUM_TAGS:
        t[c] = t[c].round(2)
    t.to_csv(cache, index=False)
    return t


# ============================================================
# slice table
# ============================================================
def _wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def _bh(p):
    p = np.asarray(p, float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    adj = np.empty(n)
    adj[order] = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(adj, 1.0)


def slice_table(df, cat_cols, num_cols, unit, min_n):
    """Failure rate of every attribute value (numeric attributes in quintiles) vs. everything else."""
    fail = ~df["correct"].astype(bool)
    N, F_ = len(df), int(fail.sum())
    base = F_ / N
    cols = {c: df[c].astype(str) for c in cat_cols if c in df}
    for c in num_cols:
        if c in df and df[c].notna().sum() > 0 and df[c].nunique() > 1:
            q = pd.qcut(df[c], 5, duplicates="drop")
            cols[f"{c} (quintile)"] = q.map(lambda iv: f"{iv.left:.4g} to {iv.right:.4g}").astype(str)
    rows = []
    for attr, vals in cols.items():
        for v, idx in vals.groupby(vals).groups.items():
            n = len(idx)
            if n < min_n or v in ("nan", "None"):
                continue
            k = int(fail.loc[idx].sum())
            rn, rk = N - n, F_ - k
            rate, rest = k / n, (rk / rn if rn else float("nan"))
            pool = F_ / N
            se = math.sqrt(pool * (1 - pool) * (1 / n + 1 / rn)) if rn and 0 < pool < 1 else 0
            z = (rate - rest) / se if se else 0.0
            lo, hi = _wilson(k, n)
            rows.append(dict(unit=unit, attribute=attr, value=v, n=n, n_fail=k, fail_rate=round(rate, 4),
                             ci_low=round(lo, 4), ci_high=round(hi, 4), rest_fail_rate=round(rest, 4),
                             lift=round(rate / rest, 3) if rest else float("nan"),
                             share_of_failures=round(k / F_, 4) if F_ else 0.0,
                             p_value=0.5 * math.erfc(z / math.sqrt(2))))
    t = pd.DataFrame(rows)
    if len(t):
        t["p_adj"] = _bh(t["p_value"])
        t = t.sort_values(["p_adj", "lift"], ascending=[True, False]).reset_index(drop=True)
    return t, base


def print_slices(t, base, title, top):
    """Worst significant value of every attribute, then the worst values overall (all of them: slices.csv)."""
    print(f"\n{title}: overall failure rate {base:.1%}")
    worse = t[(t["p_adj"] < 0.05) & (t["lift"] > 1)].sort_values("lift", ascending=False) if len(t) else t
    if not len(worse):
        print("  (no slice is significantly worse than the rest)")
        return
    print("  worst value of each attribute:")
    _print_rows(worse.drop_duplicates("attribute"))
    print(f"  worst {top} slices overall:")
    _print_rows(worse.head(top))


def _print_rows(worse):
    w = max(len(f"{a} = {v}") for a, v in zip(worse["attribute"], worse["value"]))
    print(f"  {'slice':<{w}}  {'n':>6}  {'fails':>9}  {'95% CI':>13}  {'rest':>6}  {'lift':>5}  {'% of fails':>10}")
    for r in worse.itertuples():
        print(f"  {f'{r.attribute} = {r.value}':<{w}}  {r.n:>6}  {r.fail_rate:>9.1%}  "
              f"{f'{r.ci_low:.0%}-{r.ci_high:.0%}':>13}  {r.rest_fail_rate:>6.1%}  {r.lift:>5.2f}  {r.share_of_failures:>10.1%}")


# ============================================================
# Phoenix
# ============================================================
def _json_safe(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, float) and math.isnan(v):
            v = None
        out[k] = v
    return out


def _num(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def send_to_phoenix(gt, pred, images, url, project, replace, http_timeout=60):
    import requests
    from opentelemetry import trace as ot
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from openinference.semconv.resource import ResourceAttributes

    http = requests.Session()
    try:
        r = http.get(f"{url}/v1/projects/{project}", timeout=10)
    except requests.ConnectionError:
        sys.exit(f"No Phoenix server at {url}. Start one with `phoenix serve` (or pass --url).")
    if r.status_code == 200:
        if not replace:
            sys.exit(f"Phoenix project '{project}' already exists: pass --replace to rebuild it, "
                     f"or --project <new name>.")
        http.delete(f"{url}/v1/projects/{project}", timeout=http_timeout).raise_for_status()
        print(f"Deleted existing Phoenix project '{project}'")

    provider = TracerProvider(resource=Resource({ResourceAttributes.PROJECT_NAME: project}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{url}/v1/traces"),
                                                   max_queue_size=400_000, max_export_batch_size=1000,
                                                   schedule_delay_millis=200))
    tracer = provider.get_tracer("phoenix_slices")
    sid = lambda s: format(s.get_span_context().span_id, "016x")

    def attrs(inp, out, meta):
        return {"openinference.span.kind": "CHAIN", "input.value": inp, "output.value": out,
                "metadata": json.dumps(_json_safe(meta))}

    skip_meta = {"x1", "y1", "x2", "y2", "explain", "correct", "image_id", "filepath"}
    gt_by = dict(tuple(gt.groupby("image_id"))) if len(gt) else {}
    pr_by = dict(tuple(pred.groupby("image_id"))) if len(pred) else {}
    empty = pd.DataFrame()
    ann = []          # (span_id, name, label, score, explanation)
    t = time.time_ns() - len(images) * 10_000_000
    t0 = time.time()
    for n, im in enumerate(images.to_dict("records")):
        iid = im["image_id"]
        g, p = gt_by.get(iid, empty), pr_by.get(iid, empty)
        rec, prec = _num(im.get("recall")), _num(im.get("precision"))     # NaN when the image has none
        meta = {k: v for k, v in im.items() if k not in ("filepath",)}
        root = tracer.start_span("image", start_time=t, attributes=attrs(
            im["filepath"], f"recall {'-' if rec is None else rec}, precision {'-' if prec is None else prec}: "
                            f"{im['n_detected']}/{im['n_objects']} objects found, {im['n_tp']}/{im['n_preds']} detections correct",
            meta))
        ctx = ot.set_span_in_context(root)
        q = 0
        for side, df in (("gt_object", g), ("prediction", p)):
            for u in df.to_dict("records"):
                q += 1
                st = t + q * 1000
                box = f"[{u['x1']:.0f}, {u['y1']:.0f}, {u['x2']:.0f}, {u['y2']:.0f}]"
                inp = (f"{u['category']} {box}, {u['size']} ({u['area_px']:.0f} px^2)" if side == "gt_object" else
                       f"{u['category']} {box} score {u['score']:.2f}, {u['size']} ({u['area_px']:.0f} px^2)")
                m = {k: v for k, v in u.items() if k not in skip_meta}
                m.update(image_id=iid, box_xyxy=[round(float(u[c]), 1) for c in ("x1", "y1", "x2", "y2")])
                s = tracer.start_span(side, context=ctx, start_time=st,
                                      attributes=attrs(inp, f"{u['outcome']}: {u['explain']}", m))
                s.end(end_time=st + 500)
                ok = bool(u["correct"])
                ann.append((sid(s), "correct", "correct" if ok else "incorrect", float(ok), u["explain"]))
                ann.append((sid(s), "outcome", u["outcome"], None, u["explain"]))
        root.end(end_time=t + (q + 1) * 1000 + 1000)
        if rec is not None:
            ann.append((sid(root), "recall", None, rec, f"{im['n_detected']}/{im['n_objects']} objects found"))
        if prec is not None:
            ann.append((sid(root), "precision", None, prec, f"{im['n_tp']}/{im['n_preds']} detections correct"))
        t += 10_000_000
        if (n + 1) % 500 == 0 or n + 1 == len(images):
            print(f"\r  spans: {n + 1}/{len(images)} images ({time.time() - t0:.0f}s)", end="")
    print()
    provider.force_flush()
    provider.shutdown()

    # annotations: sync=true checks the spans exist, so retry while Phoenix is still ingesting them
    print(f"  annotations: {len(ann)}")
    B = 1000
    for s in range(0, len(ann), B):
        data = [dict(span_id=a[0], name=a[1], annotator_kind="CODE",
                     result={k: v for k, v in (("label", a[2]), ("score", a[3]), ("explanation", a[4]))
                             if v is not None})
                for a in ann[s:s + B]]
        for attempt in range(300):
            r = http.post(f"{url}/v1/span_annotations", params={"sync": "true"}, json={"data": data},
                          timeout=http_timeout)
            if r.status_code != 404:
                break
            time.sleep(2)
        r.raise_for_status()
        print(f"\r  annotations: {min(s + B, len(ann))}/{len(ann)}", end="")
    print()
    pid = http.get(f"{url}/v1/projects/{project}", timeout=10).json().get("data", {}).get("id")
    return f"{url}/projects/{pid}" if pid else f"{url}/projects"


# ============================================================
def main():
    global IOU
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--preds", default=os.path.join(VAL_DIR, "predictions.json"))
    ap.add_argument("--labels", default=os.path.join(VAL_DIR, "labels.json"))
    ap.add_argument("--images", default=os.path.join(VAL_DIR, "data"))
    ap.add_argument("--models-dir", default=MODELS_DIR)
    ap.add_argument("--out", default=os.path.join(HERE, "phoenix_slices"), help="where the CSVs go")
    ap.add_argument("--iou", type=float, default=IOU)
    ap.add_argument("--conf", type=float, default=None,
                    help="drop predictions below this score first (predictions.json is already cut at 0.5)")
    ap.add_argument("--image-tags", action="store_true", help="CLIP scene tags + brightness/contrast/sharpness")
    ap.add_argument("--min-slice", type=int, default=30, help="smallest slice reported")
    ap.add_argument("--top", type=int, default=20, help="slices printed per table")
    ap.add_argument("--url", default=os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006").rstrip("/"))
    ap.add_argument("--project", default="detector-slices")
    ap.add_argument("--replace", action="store_true", help="delete the Phoenix project first if it exists")
    ap.add_argument("--no-phoenix", action="store_true", help="analysis + CSVs only")
    a = ap.parse_args()

    IOU = a.iou
    os.makedirs(a.out, exist_ok=True)
    records, cats = load(a.preds, a.labels, a.images)
    if a.conf is not None:
        for r, _, _ in records:
            r["prediction"] = [d for d in r["prediction"] if d["score"] >= a.conf]
    gt, pred, images = build_units(records, cats, a.images)

    gt_cat, gt_num, pr_cat, pr_num = list(GT_CAT), list(GT_NUM), list(PRED_CAT), list(PRED_NUM)
    if a.image_tags:
        tags = image_tags(images, a.models_dir, a.out)
        images = images.merge(tags, on="image_id", how="left")
        keep = [c for c in tags.columns if c != "image_id" and not c.endswith("_conf")]
        gt = gt.merge(tags[["image_id"] + keep], on="image_id", how="left")
        pred = pred.merge(tags[["image_id"] + keep], on="image_id", how="left")
        clip_cols = [c for c in keep if c.startswith("clip_")]
        gt_cat += clip_cols
        pr_cat += clip_cols
        gt_num += IMG_NUM_TAGS
        pr_num += IMG_NUM_TAGS

    print(f"\nGround-truth objects: {len(gt)}   " +
          ", ".join(f"{k} {v}" for k, v in gt["outcome"].value_counts().items()))
    print(f"Predictions:          {len(pred)}   " +
          ", ".join(f"{k} {v}" for k, v in pred["outcome"].value_counts().items()))

    t_gt, b_gt = slice_table(gt, gt_cat, gt_num, "gt_object", a.min_slice)
    t_pr, b_pr = slice_table(pred, pr_cat, pr_num, "prediction", a.min_slice)
    print_slices(t_gt, b_gt, "Objects the model fails to find (recall side, failure = not detected)", a.top)
    print_slices(t_pr, b_pr, "Predictions that are wrong (precision side, failure = not a true positive)", a.top)

    gt.to_csv(os.path.join(a.out, "gt_units.csv"), index=False)
    pred.to_csv(os.path.join(a.out, "pred_units.csv"), index=False)
    images.to_csv(os.path.join(a.out, "images.csv"), index=False)
    pd.concat([t_gt, t_pr]).to_csv(os.path.join(a.out, "slices.csv"), index=False)
    print(f"\nWrote gt_units.csv, pred_units.csv, images.csv, slices.csv to {a.out}")

    if a.no_phoenix:
        return
    print(f"\nSending to Phoenix at {a.url} (project '{a.project}')")
    link = send_to_phoenix(gt, pred, images, a.url, a.project, a.replace)
    print(f"\nOpen {link}  ->  Spans tab, e.g. filter:  name == 'gt_object' and metadata['size'] == 'small'")
    print("The annotation summary above the table shows the 'correct' mean (= accuracy) for the filtered slice.")


if __name__ == "__main__":
    main()
