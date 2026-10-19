"""
failure_explorer.py -- browse where and why the model fails, in the FiftyOne App.

Builds a FiftyOne dataset from the artifact folder that failure_analysis.py wrote (it never
touches the audit dataset, and works for any detector's artifacts, including demo_data.py's),
saves one view per finding, writes report.md with links into those views, and with --launch
opens the App on its own port (5154; the data audit uses 5153).

In the App
----------
ground_truth   every object, coloured by outcome (detected / missed / confused / mislocalized);
               click one: `explain` says what happened, `why` says why it was missed
model          the model's boxes at the operating point, coloured by outcome
               (tp / bkg / loc / cls / both / dupe), each with its `explain`
model_low      the model's weaker boxes (under the threshold): "it almost had it"
Views          R1.. = findings about objects the model fails to find, P1.. = wrong detections,
               O1.. = overview cuts (all misses, never proposed, scored too low, ...)
Sidebar        filter on outcome / why / findings; samples carry miss_rate, fp_rate, scene group

Run:  python failure_explorer.py --dir <artifact dir> [--images-dir <folder>] [--port 5154] [--launch] [--reuse]

Every output (dataset aside) goes into --dir. Image paths come from images.csv as the export
wrote them; under WSL, Windows paths (C:\\...) are translated to /mnt/c/... automatically, and
--images-dir points at a folder the images were moved or copied to.
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.core.utils as fou
from fiftyone import ViewField as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from failure_report import finding_codes, write_report, write_summary   # noqa: E402

PORT = 5154
GT, MODEL, LOW = "ground_truth", "model", "model_low"
SHORT_CAUSE = {"label_issue": "suspect labels", "confounded": "proxy for another attribute",
               "unannotated": "likely unannotated objects", "background_fp": "fires on background",
               "threshold_effect": "scored too low", "confusion": "class confusion", "duplicates": "duplicates",
               "localization": "bad boxes", "crowd_suppression": "suppressed by neighbours",
               "no_proposal": "never proposed", "data_sparsity": "few similar examples",
               "intrinsic": "harder as it grows"}
GT_COLORS = {"detected": "#2ca02c", "missed": "#d62728", "confused": "#ff7f0e", "mislocalized": "#e6c700",
             "ignored": "#999999"}
MODEL_COLORS = {"tp": "#2ca02c", "bkg": "#d62728", "cls": "#ff7f0e", "loc": "#e6c700", "both": "#8c564b",
                "dupe": "#9467bd"}
GT_ATTRS = ["outcome", "why", "explain", "iou", "pred_label", "pred_score", "suspect", "suspect_reason"]
PRED_ATTRS = ["outcome", "explain", "gt_label", "iou"]


def _v(x):
    """CSV value -> plain Python (NaN -> None, numpy scalars -> int/float/bool)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    if isinstance(x, np.generic):
        return x.item()
    return x


def _load(adir):
    rd = lambda n: pd.read_csv(os.path.join(adir, n), dtype={"image_id": str}, low_memory=False)
    js = lambda n, d=None: (json.load(open(os.path.join(adir, n), encoding="utf-8"))
                            if os.path.isfile(os.path.join(adir, n)) else d)
    need = ["images.csv", "gt_units.csv", "pred_units.csv", "findings.json", "finding_members.csv"]
    missing = [n for n in need if not os.path.isfile(os.path.join(adir, n))]
    if missing:
        raise SystemExit(f"Missing {missing} in {adir}: run failure_analysis.py --dir {adir} first")
    pts = None
    if os.path.isfile(os.path.join(adir, "points2d.npz")):
        z = np.load(os.path.join(adir, "points2d.npz"), allow_pickle=False)
        pts = dict(zip(z["image_ids"].astype(str), z["xy"]))
    return dict(images=rd("images.csv"), gt=rd("gt_units.csv"), pred=rd("pred_units.csv"),
                findings=js("findings.json"), members=rd("finding_members.csv"),
                clusters={c["id"]: c for c in js("clusters.json", []) or []},
                overview=js("overview.json", {}), manifest=js("manifest.json", {}) or {}, points=pts)


def _membership(members, codes):
    """{(side, image_id, idx): [codes]} for every unit in every finding's slice."""
    out = {}
    for fid, side, iid, idx in members[["finding_id", "side", "image_id", "idx"]].itertuples(index=False):
        if fid in codes:
            out.setdefault((side, iid, int(idx)), []).append(codes[fid])
    return out


def _local_path(p):
    """A path written on Windows ("C:\\...") as this machine can open it: /mnt/c/... under WSL/Linux."""
    p = str(p)
    if os.name != "nt" and len(p) > 2 and p[1] == ":" and p[2] in "\\/":
        return f"/mnt/{p[0].lower()}/" + p[3:].replace("\\", "/")
    return p


def resolve_images(images, images_dir=None):
    """Point every image at a file that exists HERE. With images_dir, each image is looked up by
    file name in that folder (images moved or copied); otherwise the exported path is used, with
    Windows paths translated for WSL. Images that can't be found are dropped from the App (the
    analysis and the reports still count them)."""
    names = images["filepath"].astype(str).str.replace("\\", "/").str.split("/").str[-1]
    if images_dir:
        paths = [os.path.join(images_dir, n) for n in names]
    else:
        paths = [_local_path(p) for p in images["filepath"]]
    found = np.array([os.path.isfile(p) for p in paths])
    if not found.any():
        raise SystemExit(f"None of the {len(images)} images were found "
                         f"({'in ' + images_dir if images_dir else 'e.g. ' + paths[0]}). "
                         "Pass --images-dir <folder holding the image files>.")
    if not found.all():
        print(f"WARNING: {int((~found).sum())} of {len(images)} images not found "
              f"({'in ' + images_dir if images_dir else 'at their exported paths'}); they are left out of the App. "
              "The reports still cover all of them.")
    out = images.copy()
    out["filepath"] = paths
    return out[found].reset_index(drop=True)


def build_dataset(name, art, low_min, images_dir=None):
    images, gt, pred = resolve_images(art["images"], images_dir), art["gt"], art["pred"]
    codes = finding_codes(art["findings"])
    member = _membership(art["members"], codes)
    clusters = art["clusters"]
    thr = art["overview"].get("config", {}).get("conf_thresh", 0.5)

    if name in fo.list_datasets():
        fo.delete_dataset(name)
    ds = fo.Dataset(name, persistent=True)

    gt_by = {k: v for k, v in gt.groupby("image_id")}
    op = pred[pred["outcome"] != "subthr"]
    low = pred[(pred["outcome"] == "subthr") & (pred["score"] >= low_min)]
    op_by = {k: v for k, v in op.groupby("image_id")}
    low_by = {k: v for k, v in low.groupby("image_id")}
    empty = pd.DataFrame()
    skip = {"image_id", "filepath", "width", "height", "issues"}
    img_cols = [c for c in images.columns if c not in skip]

    samples = []
    for r in images.to_dict("records"):
        iid = r["image_id"]
        g = gt_by.get(iid, empty)
        p = op_by.get(iid, empty)
        lw = low_by.get(iid, empty)
        gdets, pdets, ldets, img_codes = [], [], [], set()
        for u in g.to_dict("records"):
            fl = member.get(("recall", iid, int(u["gt_idx"])), [])
            img_codes.update(fl)
            gdets.append(fo.Detection(label=str(u["label"]), bounding_box=[u["x"], u["y"], u["w"], u["h"]],
                                      gt_idx=int(u["gt_idx"]), findings=fl, crowd=bool(u["crowd"]),
                                      **{a: _v(u.get(a)) for a in GT_ATTRS if a in u}))
        for u in p.to_dict("records"):
            fl = member.get(("precision", iid, int(u["pred_idx"])), [])
            img_codes.update(fl)
            pdets.append(fo.Detection(label=str(u["label"]), bounding_box=[u["x"], u["y"], u["w"], u["h"]],
                                      confidence=float(u["score"]), pred_idx=int(u["pred_idx"]), findings=fl,
                                      **{a: _v(u.get(a)) for a in PRED_ATTRS if a in u}))
        for u in lw.to_dict("records"):
            ldets.append(fo.Detection(label=str(u["label"]), bounding_box=[u["x"], u["y"], u["w"], u["h"]],
                                      confidence=float(u["score"]), pred_idx=int(u["pred_idx"])))
        scored = [d for d in gdets if not d["crowd"]]
        n_missed = sum(d["outcome"] != "detected" for d in scored)
        n_fp = sum(d["outcome"] != "tp" for d in pdets)
        sc = next((x for x in (g.get("scene_cluster"), p.get("scene_cluster")) if x is not None and len(x)), None)
        sc = _v(sc.iloc[0]) if sc is not None else None
        s = fo.Sample(filepath=r["filepath"], source_id=iid,
                      metadata=fo.ImageMetadata(width=int(r["width"]), height=int(r["height"])),
                      **{GT: fo.Detections(detections=gdets), MODEL: fo.Detections(detections=pdets),
                         LOW: fo.Detections(detections=ldets)},
                      n_gt=len(scored), n_missed=n_missed, miss_rate=(n_missed / len(scored) if scored else None),
                      n_pred=len(pdets), n_fp=n_fp, fp_rate=(n_fp / len(pdets) if pdets else None),
                      findings=sorted(img_codes, key=lambda c: (c[0], int(c[1:]))),
                      scene_cluster=sc, scene_desc=(clusters.get(sc, {}).get("label") if sc else None),
                      issues=[t for t in str(_v(r.get("issues")) or "").split(";") if t],
                      **{c: _v(r[c]) for c in img_cols})
        samples.append(s)
    ds.add_samples(samples)
    ds.add_dynamic_sample_fields()          # outcome / why / findings ... become sidebar filters
    ds.info.update(findings_generated=art["findings"].get("generated"), source=art["manifest"],
                   conf_thresh=thr, low_min=low_min, images_dir=images_dir)
    ds.save()
    print(f"Built '{name}': {len(ds)} images, {ds.count(f'{GT}.detections')} objects, "
          f"{ds.count(f'{MODEL}.detections')} detections at score >= {thr}, "
          f"{ds.count(f'{LOW}.detections')} weaker ones (>= {low_min})")
    return ds


def save_views(ds, findings, has_suspect):
    """One saved view per finding (only its failed units) plus overview cuts. -> {key: view name}"""
    for v in ds.list_saved_views():
        ds.delete_saved_view(v)
    names = {}

    def put(key, name, view, desc):
        ds.save_view(name, view, description=desc[:1000])
        names[key] = name

    codes = finding_codes(findings)
    for side, field, ok, sort in (("recall", GT, "detected", "n_missed"), ("precision", MODEL, "tp", "n_fp")):
        for c in findings[side]:
            code = codes[c["id"]]
            sup = [SHORT_CAUSE.get(h["key"], h["key"]) for h in c["hypotheses"] if h["verdict"] == "supports"][:2]
            name = f"{code} · {c['label']} — {', '.join(sup) or 'inconclusive'}"
            view = ds.filter_labels(field, F("findings").contains(code) & (F("outcome") != ok)).sort_by(sort, reverse=True)
            put(c["id"], name, view, c["headline"])

    failed = ["missed", "confused", "mislocalized"]
    put("O1", "O1 · All objects the model failed on",
        ds.filter_labels(GT, F("outcome").is_in(failed)).sort_by("n_missed", reverse=True),
        "Every ground-truth object not detected, worst images first.")
    put("O2", "O2 · Missed: never proposed", ds.filter_labels(GT, F("why") == "no_proposal"),
        "No prediction overlaps these objects at any score: a recall ceiling.")
    put("O3", "O3 · Missed: found but scored under the threshold",
        ds.filter_labels(GT, F("why") == "low_score_correct"),
        f"A correct box exists in '{LOW}' but scored under the threshold.")
    put("O4", "O4 · Found, but wrong class", ds.filter_labels(GT, F("outcome") == "confused"),
        "The model's box is in the right place but has the wrong class.")
    put("O5", "O5 · Confident detections on background (possibly unannotated)",
        ds.filter_labels(MODEL, (F("outcome") == "bkg") & (F("confidence") >= 0.8)),
        "High-scoring boxes with no ground truth under them: often objects missing from the labels.")
    if has_suspect:
        put("O6", "O6 · Suspect labels (an independent model disagrees)", ds.filter_labels(GT, F("suspect") == True),
            "Boxes flagged upstream as likely wrong labels; failures here may not be the model's fault.")
    print(f"Saved {len(names)} views")
    return names


def color_and_layout(ds, points):
    try:
        ds.app_config.color_scheme = fo.ColorScheme(color_by="value", fields=[
            dict(path=GT, colorByAttribute="outcome", valueColors=[dict(value=k, color=c) for k, c in GT_COLORS.items()]),
            dict(path=MODEL, colorByAttribute="outcome",
                 valueColors=[dict(value=k, color=c) for k, c in MODEL_COLORS.items()]),
            dict(path=LOW, fieldColor="#7f7f7f")])
        ds.save()
    except Exception as e:
        print(f"(colour scheme skipped: {e})")
    side = [fo.Panel(type="Histograms")]
    if points:
        ids = ds.values("source_id")
        view = ds.select([sid for sid, iid in zip(ds.values("id"), ids) if iid in points])
        xy = np.array([points[i] for i in view.values("source_id")], float)
        fob.compute_visualization(view, points=xy, brain_key="map")
        side.insert(0, fo.Panel(type="Embeddings", state=dict(brainKey="map", colorByField="miss_rate")))
    try:
        ws = fo.Space(children=[fo.Space(children=[fo.Panel(type="Samples", pinned=True)]), fo.Space(children=side)],
                      orientation="horizontal")
        ds.save_workspace("Model Explain", ws, description="Samples + failure map + histograms", overwrite=True)
    except Exception as e:
        print(f"(workspace skipped: {e})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dir", required=True, help="artifact folder (after failure_analysis.py)")
    ap.add_argument("--name", default=None, help="FiftyOne dataset name (default: explain-<dir name>)")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--low-min", type=float, default=0.1, help="weakest sub-threshold box kept in model_low")
    ap.add_argument("--reuse", action="store_true", help="keep the dataset if it was built from these findings")
    ap.add_argument("--launch", action=argparse.BooleanOptionalAction, default=False,
                    help="open the FiftyOne App when done (default: build + report only)")
    ap.add_argument("--images-dir", default=None,
                    help="folder holding the image files, if they moved since the export (looked up by file name)")
    a = ap.parse_args()
    adir = os.path.abspath(a.dir)
    name = a.name or "explain-" + os.path.basename(adir.rstrip("\\/"))
    art = _load(adir)
    images_dir = os.path.abspath(os.path.expanduser(a.images_dir)) if a.images_dir else None

    ds = fo.load_dataset(name) if name in fo.list_datasets() else None
    if not (a.reuse and ds is not None and ds.info.get("findings_generated") == art["findings"].get("generated")
            and ds.info.get("images_dir") == images_dir):
        ds = build_dataset(name, art, a.low_min, images_dir)
        names = save_views(ds, art["findings"], bool(art["gt"]["suspect"].fillna(False).astype(bool).any()))
        color_and_layout(ds, art["points"])
        ds.info["view_names"] = names
        ds.save()
    else:
        print(f"Reusing '{name}' (built from these findings)")
        names = ds.info.get("view_names", {})

    base = f"http://localhost:{a.port}/datasets/{name}"
    links = {k: f"{base}?view={fou.to_slug(v)}" for k, v in names.items()}
    links["app"] = base
    write_report(adir, links)
    write_summary(adir, links)

    if not a.launch:
        print(f"Reports: {os.path.join(adir, 'report_summary.md')} (short), {os.path.join(adir, 'report.md')} (full)")
        print("App not opened (pass --launch to browse the dataset)")
        return
    first = next((v for k, v in names.items() if k.startswith("recall:")), None)
    session = fo.launch_app(ds, view=ds.load_saved_view(first) if first else None, port=a.port)
    print(f"\nApp running at {base}  (views: R = not found, P = wrong detections, O = overview)")
    print(f"Reports: {os.path.join(adir, 'report_summary.md')} (short), {os.path.join(adir, 'report.md')} (full)")
    session.wait()


if __name__ == "__main__":
    main()
