"""
FiftyOne data-audit workflow
=============================

Three data-auditing jobs, all made point-and-click in the FiftyOne App
(http://localhost:5151):

  A. QUALITY   -> which images are good enough to bother annotating
                  (measured by the @jacobmarks/image_issues plugin: blurriness,
                  brightness, contrast, saturation, exposure, entropy, aspect ratio,
                  salt-and-pepper noise, vignetting)
  B. DUPLICATES -> near-duplicate images you shouldn't pay to annotate twice
  C. ANNOTATIONS -> existing labels that are likely missing / wrong / spurious

Everything is computed once, then persisted as named **Saved Views** (visible in the
App's view dropdown) and a saved **Workspace** (Samples + Embeddings + Histograms +
Model-Evaluation panels). The App is launched a single time and kept alive with
`session.wait()`.

It also exports the plain-file artifacts explainability/failure_analysis.py reads, to
explain where and why the model under test (MODEL_FIELD) fails; GT boxes an independent
model (SUSPECT_TAG) confidently disagrees with are flagged as suspect labels.

Run from a directory where the file is NOT named `fiftyone.py` (that shadows the
package). With --launch, then open http://localhost:5153 (see APP_PORT).

    python fiftyone_data_audit.py                     # full audit + export (no App)
    python fiftyone_data_audit.py --launch            # full audit + export, then open the App
    python fiftyone_data_audit.py --export-explain    # export only, from the existing dataset
    python fiftyone_data_audit.py --evaluate-planted  # test set: report planted errors caught
    python fiftyone_data_audit.py --export-explain --evaluate-planted --include-planted --explain-dir <dir>

Paths default to the CONFIG section below; override them per run, e.g. for another dataset:
    python fiftyone_data_audit.py --dataset-name bccd-audit --data-path <images> --labels-path <labels.json>
        --csv-path <scores.csv> --model-predictions <preds.json> --suspect-predictions <preds.json>
        --suspect-tag yolo11n --explain-dir <dir>
"""

import os
import csv
import json
import time
import importlib.util
from collections import Counter
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageFile

import torch
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob
import fiftyone.core.utils as fou
from fiftyone import ViewField as F

# let truncated / corrupt files raise so we can flag them as low quality
ImageFile.LOAD_TRUNCATED_IMAGES = False


# ============================================================
# 0. CONFIG  (edit these to point at your own data)
# ============================================================
DATASET_NAME = "data-audit"

# Load images + labels from disk. Set DATA_PATH=None to fall back to the COCO zoo.
VALID_DIR       = r"C:/Users/Gabriel/fiftyone/coco-2017/validation"
DATA_PATH       = os.path.join(VALID_DIR, "data")           # folder of images
LABELS_PATH     = os.path.join(VALID_DIR, "labels.json")    # COCO labels to AUDIT
# Planted test errors are read from labels.json itself (see mark_planted): no separate answer key.

GT_FIELD   = "ground_truth"   # where COCO boxes land
PRED_FIELD = "predictions"    # detector's independent 2nd opinion

# Image quality comes from the @jacobmarks/image_issues plugin
# (https://github.com/jacobmarks/image-quality-issues): its 9 measurements and its
# issue definitions/thresholds are the source of truth. This script only adds the
# two checks the plugin lacks: unreadable files and images that are too small.
IMAGE_ISSUES_PLUGIN = os.path.join(
    fo.config.plugins_dir, "@jacobmarks", "image_issues", "__init__.py"
)
# Override any plugin default threshold here, e.g. {"blurry": 80.0, "dark": 0.08}.
# Issues: bright, dark, weird_aspect_ratio, blurry, low_entropy, low_exposure,
# high_exposure, low_contrast, high_contrast, low_saturation, high_saturation
ISSUE_THRESHOLDS = {}
MIN_SIDE = 100          # px on the short edge; smaller is flagged "tiny"

# Which issues make an image NOT worth annotating (-> low_quality). Every issue is
# still flagged and searchable; these are just the ones that reject. Only real
# defects reject by default. On COCO, at the plugin's default thresholds, the others
# fire on ordinary photos: high_contrast ~99%, low_exposure ~63%, bright ~18%,
# high_exposure ~14%, low_saturation ~13%. weird_aspect_ratio (panoramas) is
# still annotatable.
REJECT_ISSUES = {"corrupt", "tiny", "blurry", "dark", "low_entropy", "low_contrast"}
QUALITY_WORKERS = 8     # threads for the per-image measurements (OpenCV releases the GIL)

# Duplicate threshold (cosine distance in CLIP space; smaller = stricter)
DUP_THRESH = 0.2

# How many images you can afford to annotate (curation budget)
BUDGET = 1000

# GPU when available (this step is the slow one)
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
# Keep FULL precision: torchvision's Faster R-CNN silently produces ZERO
# detections in FP16 (the RPN/box-coder math breaks), which would make every GT
# box look "missing" in the annotation audit. Full-precision GPU is still ~120
# img/s here, so there's no reason to risk half precision.
USE_HALF = False
# Windows + torch DataLoader workers need the __main__ guard (we have one). Keep 0 on
# CPU (avoids spawn overhead when compute-bound); use a few workers on GPU to feed it.
NUM_WORKERS = 4 if DEVICE == "cuda" else 0
BATCH_SIZE  = 8

# Where to dump the per-image score table (one row per image, all categories)
CSV_PATH = r"C:/Users/Gabriel/Documents/audit_scores.csv"

# App port. Not the default 5151: any other live FiftyOne session on 5151 (e.g. a
# Jupyter kernel) pushes its own dataset to every tab on that port.
APP_PORT = 5153

# --- Model explainability export (explainability/failure_analysis.py reads it) ---
# The model being EXPLAINED. pred_rcnn has the same Faster R-CNN weights as PRED_FIELD but
# keeps scores down to 0.01 (PRED_FIELD stops at 0.3): the analysis needs the low-scoring
# boxes to tell "never proposed" apart from "proposed but scored under the threshold".
MODEL_FIELD       = "pred_rcnn"
MODEL_PREDICTIONS = os.path.join(VALID_DIR, "predictions_rcnn.json")   # loaded if MODEL_FIELD is absent
# An INDEPENDENT model decides which GT boxes are suspect labels. It must not be the model
# being explained, or "the label is wrong" would just mean "the model disagrees".
SUSPECT_TAG         = "yolo11x"
SUSPECT_PREDICTIONS = os.path.join(VALID_DIR, "predictions_yolo11x.json")
# A GT box is suspect only when the suspect model is confident about it:
#   class_disagree: its box there has another class; mistakenness = (conf + 1) / 2, so 0.8 = conf >= 0.6
#   spurious:       it proposes nothing of any class near a visible, unoccluded box
SUSPECT_CLASS_MIN   = 0.80
SPURIOUS_MAX_IOU    = 0.1       # best IoU of ANY suspect-model box (any class, any score) must be below this
SPURIOUS_MIN_AREA   = 32 ** 2   # px; smaller boxes are too often missed by every detector
SPURIOUS_MAX_COVER  = 0.5       # skip boxes this much covered by other GT boxes
EXPLAIN_DIR = os.path.join(VALID_DIR, "explainability", "artifacts_rcnn")
# With --evaluate-planted, the planted test copies (_wrong, _duplicate, _missing) are recognised
# and left out of the export (they distort a model diagnosis); include them only to validate the
# explainability pipeline against their answer keys. Without it, every image is exported.
EXPLAIN_INCLUDE_PLANTED = False


# ============================================================
# 1. LOAD DATASET (from disk, idempotent)
# ============================================================
def load_dataset():
    if DATASET_NAME in fo.list_datasets():
        print(f"Reusing existing dataset '{DATASET_NAME}'")
        return fo.load_dataset(DATASET_NAME)

    if DATA_PATH and os.path.isdir(DATA_PATH):
        print(f"Importing COCO data from {DATA_PATH}")
        ds = fo.Dataset.from_dir(
            dataset_type=fo.types.COCODetectionDataset,
            data_path=DATA_PATH,
            labels_path=LABELS_PATH,
            label_field=GT_FIELD,
            label_types=["detections"],   # boxes only; COCO would also add a *_segmentations field
            name=DATASET_NAME,
        )
    else:
        print("DATA_PATH not found; falling back to the COCO-2017 zoo split")
        ds = foz.load_zoo_dataset(
            "coco-2017", split="validation", dataset_name=DATASET_NAME
        )

    ds.persistent = True
    ds.compute_metadata()  # width/height/size on disk
    return ds


def gt_fields(dataset):
    """COCO import can name the boxes field `ground_truth` or `ground_truth_detections`
    (it suffixes with the label type), and both can exist: blackbox_fiftyone.add_new_images
    imports later images under the unsuffixed name. Existing ones, most boxes first."""
    schema = dataset.get_field_schema()
    names = [n for n in (f"{GT_FIELD}_detections", GT_FIELD) if n in schema]
    return sorted(names, key=lambda n: -dataset.count(f"{n}.detections"))


def resolve_gt_field(dataset):
    """The main GT field: the one holding the most boxes."""
    return (gt_fields(dataset) or [GT_FIELD])[0]


# ============================================================
# 2. EMBEDDINGS + SIMILARITY + VISUALIZATION  (compute once)
# ============================================================
def compute_embeddings(dataset):
    runs = dataset.list_brain_runs()

    if "clip_sim" not in runs:
        print("Computing CLIP embeddings + similarity index (clip_sim)...")
        fob.compute_similarity(
            dataset,
            model="clip-vit-base32-torch",
            embeddings="clip_emb",   # stored per-sample, reused everywhere (no re-inference)
            brain_key="clip_sim",
        )
    else:
        print("Reusing similarity index 'clip_sim'")

    if "clip_viz" not in dataset.list_brain_runs():
        print("Computing 2D visualization for the Embeddings panel (clip_viz)...")
        try:
            fob.compute_visualization(
                dataset, embeddings="clip_emb", brain_key="clip_viz", method="umap"
            )
        except Exception:
            # umap-learn not installed -> t-SNE (sklearn) works out of the box
            fob.compute_visualization(
                dataset, embeddings="clip_emb", brain_key="clip_viz", method="tsne"
            )
    else:
        print("Reusing visualization 'clip_viz'")


# ============================================================
# 3. TASK A -- QUALITY GATE  (which images are usable, and WHY)
# ============================================================
# Per-image measurements written by the plugin (exposure splits into min/max)
QUALITY_PROPS = ["aspect_ratio", "blurriness", "brightness", "contrast", "entropy",
                 "min_exposure", "max_exposure", "salt_and_pepper", "saturation",
                 "vignetting"]
# Fields from this script's old home-grown quality check. `brightness`/`contrast`
# reuse the plugin's names on a DIFFERENT scale, and the plugin skips any field that
# already exists, so they must be dropped before the plugin values are written.
LEGACY_QUALITY_FIELDS = ["blur", "brightness", "contrast", "is_corrupt"]
OWN_ISSUES = ["corrupt", "tiny"]   # the only checks the plugin doesn't cover


def load_image_issues_plugin():
    if not os.path.isfile(IMAGE_ISSUES_PLUGIN):
        raise RuntimeError(
            f"image_issues plugin not found at {IMAGE_ISSUES_PLUGIN}. Install it with:\n"
            "  fiftyone plugins download https://github.com/jacobmarks/image-quality-issues/"
        )
    spec = importlib.util.spec_from_file_location("image_issues", IMAGE_ISSUES_PLUGIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def issue_rules(plugin):
    """{issue: (property, threshold, flag_if_less_than)} from the plugin's own
    ISSUE_MAPPING, with any ISSUE_THRESHOLDS overrides applied."""
    return {name: (m["base_property"], ISSUE_THRESHOLDS.get(name, m["threshold"]), m["lt"])
            for name, m in plugin.ISSUE_MAPPING.items()}


def _measure_image(plugin, sample):
    """Readability check, then every plugin measurement for one image."""
    try:
        with Image.open(sample.filepath) as im:
            im.load()                      # full decode: truncated files raise here
    except Exception:
        return True, {p: None for p in QUALITY_PROPS}

    props = {}
    for prop, fn in plugin.PROP_SAMPLE_COMPUTE_FUNCTIONS.items():
        try:
            with np.errstate(divide="ignore", invalid="ignore"):   # vignetting /0 on dark centers
                v = fn(sample)
        except Exception:
            v = (None, None) if prop == "exposure" else None
        if prop == "exposure":
            props["min_exposure"], props["max_exposure"] = v
        else:
            props[prop] = v
    # numpy scalars -> float; NaN (e.g. vignetting on a flat image) -> None
    props = {k: (None if v is None or not np.isfinite(v) else float(v)) for k, v in props.items()}
    return False, props


def audit_quality(dataset):
    print("\n=== TASK A: QUALITY GATE (image_issues plugin) ===")
    plugin = load_image_issues_plugin()
    rules = issue_rules(plugin)
    all_issues = OWN_ISSUES + list(rules)

    # 1) measurements -- expensive, so only when missing or still on the old scheme
    schema = dataset.get_field_schema()
    migrated = dataset.info.get("quality_backend") == "image_issues"
    if not migrated:
        old = [f for f in LEGACY_QUALITY_FIELDS if f in schema]
        if old:
            print(f"Dropping old quality fields replaced by the plugin: {old}")
            dataset.delete_sample_fields(old)
        schema = dataset.get_field_schema()

    if migrated and all(p in schema for p in QUALITY_PROPS + ["corrupt"]):
        print("Reusing image_issues measurements")
    else:
        samples = list(dataset.select_fields("metadata"))
        print(f"Measuring {len(samples)} images with the image_issues plugin "
              f"({QUALITY_WORKERS} threads)...")
        with ThreadPoolExecutor(QUALITY_WORKERS) as ex, \
                fou.ProgressBar(total=len(samples)) as pb:
            results = list(pb(ex.map(lambda s: _measure_image(plugin, s), samples)))
        dataset.set_values("corrupt", [c for c, _ in results])
        for prop in QUALITY_PROPS:
            dataset.set_values(prop, [p[prop] for _, p in results])
        dataset.info["quality_backend"] = "image_issues"
        dataset.save()

    # 2) issue flags: one bool field per issue (same names the plugin's find_issues
    #    operator uses), plus the plugin's tagging convention (<issue> + "issue")
    widths  = dataset.values("metadata.width")
    heights = dataset.values("metadata.height")
    min_side = [min(w, h) if (w and h) else 0 for w, h in zip(widths, heights)]
    dataset.set_values("min_side", min_side)

    corrupt = np.array(dataset.values("corrupt"), dtype=bool)
    flags = {"corrupt": corrupt,
             "tiny": ~corrupt & (np.array(min_side) < MIN_SIDE)}
    for name, (prop, thresh, lt) in rules.items():
        vals = np.array([np.nan if v is None else v for v in dataset.values(prop)], dtype=float)
        with np.errstate(invalid="ignore"):
            flags[name] = (vals < thresh) if lt else (vals > thresh)   # NaN -> False

    dataset.untag_samples(all_issues + ["issue", "overexposed"])   # tags from earlier runs
    ids = np.array(dataset.values("id"))
    for name in all_issues:
        dataset.set_values(name, flags[name].tolist())
        if flags[name].any():
            dataset.select(ids[flags[name]].tolist()).tag_samples([name, "issue"])

    # 3) summaries: `issues` lists EVERY issue an image has (a list field, so the
    #    sidebar shows each issue with a count, click to filter); `quality` names only
    #    the issues that reject it; `low_quality` is the keep/drop gate.
    per_image = [[n for n in all_issues if flags[n][i]] for i in range(len(ids))]
    rejecting = [[n for n in issues if n in REJECT_ISSUES] for issues in per_image]
    dataset.set_values("issues", per_image)
    dataset.set_values("n_issues", [len(x) for x in per_image])
    dataset.set_values("quality", ["ok" if not r else "+".join(r) for r in rejecting])
    dataset.set_values("low_quality", [bool(r) for r in rejecting])

    # read-out to help TUNE thresholds (ISSUE_THRESHOLDS / REJECT_ISSUES above)
    print(f"{'issue':20s} {'rule':32s} {'images':>7s}  rejects?")
    print(f"{'corrupt':20s} {'unreadable file':32s} {flags['corrupt'].sum():7d}  "
          f"{'yes' if 'corrupt' in REJECT_ISSUES else 'no'}")
    print(f"{'tiny':20s} {f'min_side < {MIN_SIDE}px':32s} {flags['tiny'].sum():7d}  "
          f"{'yes' if 'tiny' in REJECT_ISSUES else 'no'}")
    for name, (prop, thresh, lt) in rules.items():
        rule = f"{prop} {'<' if lt else '>'} {thresh:g}"
        print(f"{name:20s} {rule:32s} {flags[name].sum():7d}  "
              f"{'yes' if name in REJECT_ISSUES else 'no'}")
    print("measurement p5 / p50 / p95:")
    for prop in QUALITY_PROPS:
        v = np.array([x for x in dataset.values(prop) if x is not None], dtype=float)
        if len(v):
            p5, p50, p95 = np.percentile(v, [5, 50, 95])
            print(f"  {prop:16s} {p5:10.3f} {p50:10.3f} {p95:10.3f}")
    print(f"low_quality total: {sum(bool(r) for r in rejecting)} / {len(dataset)}")
    return rules


# ============================================================
# 4. TASK B -- DUPLICATES / NEAR-DUPLICATES
# ============================================================
def audit_duplicates(dataset):
    print("\n=== TASK B: DUPLICATES ===")
    sim = dataset.load_brain_results("clip_sim")

    # near-duplicates you shouldn't pay to annotate twice
    sim.find_duplicates(thresh=DUP_THRESH)
    dup_ids = list(sim.duplicate_ids)
    print(f"Near-duplicate images (redundant) at thresh={DUP_THRESH}: {len(dup_ids)}")

    # Capture the duplicates view NOW: find_unique() below overwrites the index's
    # internal state, after which sim.duplicates_view() raises "must first call
    # find_duplicates()". The returned view bakes in concrete sample IDs, so it
    # stays valid to save even after the index state moves on.
    try:
        dup_view = sim.duplicates_view()
    except Exception as e:
        print(f"(duplicates_view unavailable: {e})")
        dup_view = None

    # bool flag -> clickable in the sidebar
    is_dup = [False] * len(dataset)
    id_to_idx = {sid: i for i, sid in enumerate(dataset.values("id"))}
    for sid in dup_ids:
        if sid in id_to_idx:
            is_dup[id_to_idx[sid]] = True
    dataset.set_values("is_duplicate", is_dup)

    # per-sample nearest-neighbor cosine distance -> RANGE SLIDER to tune the
    # near-dup threshold live (small distance == likely duplicate). NOTE: changing
    # DUP_THRESH itself and re-flagging still requires re-running this script.
    embs = np.asarray(dataset.values("clip_emb"), dtype=float)
    try:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=2, metric="cosine").fit(embs)
        dist, _ = nn.kneighbors(embs)      # col 0 == self (dist 0), col 1 == nearest other
        dataset.set_values("nn_distance", [float(d) for d in dist[:, 1]])
    except Exception as e:
        print(f"(nn_distance slider unavailable: {e})")

    # a maximally-diverse subset chosen jointly by the index (for the budget)
    sim.find_unique(min(BUDGET, len(dataset)))
    try:
        uniq_view = sim.unique_view()
    except Exception as e:
        print(f"(unique_view unavailable: {e})")
        uniq_view = None
    return sim, dup_ids, dup_view, uniq_view


# ============================================================
# 5. TASK C -- ANNOTATION-ERROR AUDIT
#    (use an independent detector as a 2nd opinion; run on ALL images, GPU-aware)
# ============================================================
def audit_annotations(dataset, gt_field):
    print("\n=== TASK C: ANNOTATION ERRORS ===")
    schema = dataset.get_field_schema()
    # count() raises on a nonexistent field, so check the schema first
    if gt_field not in schema or dataset.count(f"{gt_field}.detections") == 0:
        print(f"No '{gt_field}' labels present; skipping annotation audit.")
        return False

    # 1) detector predictions (idempotent, but recompute if a prior run only
    #    partially populated predictions, e.g. an interrupted CPU run). NOTE:
    #    an interrupted run leaves an EMPTY `predictions` field on every sample
    #    (non-null, 0 boxes), which exists().count() would wrongly treat as
    #    "done" -> the eval then sees no predictions and marks all GT as missed.
    #    Gate on whether any boxes were actually predicted instead.
    n_pred_boxes = dataset.count(f"{PRED_FIELD}.detections") if PRED_FIELD in schema else 0
    need_preds = n_pred_boxes == 0
    if need_preds:
        try:
            model = foz.load_zoo_model(
                "faster-rcnn-resnet50-fpn-coco-torch",
                device=DEVICE, use_half_precision=USE_HALF,
            )
        except Exception:
            model = foz.load_zoo_model("faster-rcnn-resnet50-fpn-coco-torch")
        print(f"Applying detector on {len(dataset)} images (device={DEVICE}) -- this is the slow step...")
        dataset.apply_model(
            model, label_field=PRED_FIELD,
            batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, skip_failures=True,
        )
    else:
        print(f"'{PRED_FIELD}' already present; skipping apply_model")

    # 2) COCO-style TP/FP/FN match, stored per-detection + per-sample (drives the
    #    Model-Evaluation panel: confusion matrix, per-class P/R, click-through)
    if "eval" in dataset.list_evaluations():
        dataset.delete_evaluation("eval")
    dataset.evaluate_detections(PRED_FIELD, gt_field=gt_field, eval_key="eval")

    # 3) brain: how likely each GT label is wrong + which boxes are missing / spurious
    fob.compute_mistakenness(dataset, PRED_FIELD, label_field=gt_field)

    print(f"FP predictions (model sees, GT doesn't) : {dataset.sum('eval_fp')}")
    print(f"FN GT       (GT has, model misses)      : {dataset.sum('eval_fn')}")
    print(f"images w/ possible MISSING annotation   : {dataset.match(F('possible_missing') > 0).count()}")
    print(f"images w/ possible SPURIOUS annotation  : {dataset.match(F('possible_spurious') > 0).count()}")
    mm = dataset.bounds("mistakenness")
    if mm[1] is not None:
        print(f"max GT-label mistakenness (0-1)         : {mm[1]:.3f}")
    return True


def mark_planted(dataset, gt_field):
    """OPTIONAL: the testbed plants errors in labels.json itself. "<name>_wrong.jpg" is a
    copy of "<name>.jpg" whose boxes had their class changed; "<name>_duplicate.jpg" is an
    exact copy. Tag them (sample field `planted`, GT detection attribute `planted`) so you
    can confirm the audit surfaces them."""
    names = [os.path.basename(fp) for fp in dataset.values("filepath")]
    index = {n: i for i, n in enumerate(names)}
    originals, kinds = [], []
    for n in names:
        stem, ext = os.path.splitext(n)
        orig, kind = None, None
        for suffix, k in (("_wrong", "wrong"), ("_duplicate", "duplicate")):
            if stem.endswith(suffix) and stem[:-len(suffix)] + ext in index:
                orig, kind = index[stem[:-len(suffix)] + ext], k
        originals.append(orig)
        kinds.append(kind)
    if not any(kinds):
        return False
    print("\n=== PLANTED TEST ERRORS (from labels.json) ===")
    dataset.set_values("planted", kinds)

    # a box in a _wrong image is planted if the same box in the original has another class
    boxes = dataset.values(f"{gt_field}.detections.bounding_box")
    classes = dataset.values(f"{gt_field}.detections.label")
    key = lambda b: tuple(round(x, 6) for x in b)
    planted_dets = []
    for i in range(len(names)):
        b, c = boxes[i] or [], classes[i] or []
        if kinds[i] != "wrong":
            planted_dets.append([False] * len(b))
            continue
        j = originals[i]
        orig = {key(bb): cc for bb, cc in zip(boxes[j] or [], classes[j] or [])}
        planted_dets.append([orig.get(key(bb)) != cc for bb, cc in zip(b, c)])
    has_gt = [b is not None for b in boxes]
    dataset.exists(gt_field).set_values(
        f"{gt_field}.detections.planted", [p for p, h in zip(planted_dets, has_gt) if h])

    n_planted = sum(map(sum, planted_dets))
    print(f"{kinds.count('wrong')} _wrong images ({n_planted} wrong-class boxes), "
          f"{kinds.count('duplicate')} _duplicate images")

    # how much of it the audit catches
    schema = dataset.get_field_schema(flat=True)
    if "is_duplicate" in schema:
        # caught if either copy is flagged (find_duplicates may keep the _duplicate as representative)
        is_dup = dataset.values("is_duplicate")
        pairs = [(i, j) for i, (k, j) in enumerate(zip(kinds, originals)) if k == "duplicate"]
        caught = sum(bool(is_dup[i] or is_dup[j]) for i, j in pairs)
        print(f"  _duplicate pairs flagged is_duplicate  : {caught} / {len(pairs)}")
    if "mistakenness" in schema:  # per-box attributes aren't declared, so check the sample field
        m = dataset.values(f"{gt_field}.detections.mistakenness")
        ranked = sorted(((mm, pp) for ms, ps in zip(m, planted_dets) for mm, pp in zip(ms or [], ps)
                         if mm is not None), key=lambda x: -x[0])
        n_scored = sum(p for _, p in ranked)
        print(f"  planted boxes with a mistakenness score : {n_scored} / {n_planted}")
        for k in (500, 1000, 2000, 5000):
            hit = sum(p for _, p in ranked[:k])
            print(f"  top {k:5d} GT boxes by mistakenness   : {hit:4d} planted "
                  f"({hit / max(n_planted, 1):4.0%} recall, {hit / k:4.0%} precision)")
    if "possible_spurious" in schema:
        wrong = dataset.match(F("planted") == "wrong")
        print(f"  _wrong images with possible_spurious > 0: {wrong.match(F('possible_spurious') > 0).count()} / {len(wrong)}")
    return True


# ============================================================
# 6. CURATION SHORTLIST  (redundancy + value)
# ============================================================
def build_shortlist(dataset, dup_ids):
    print("\n=== CURATION SHORTLIST ===")
    if "uniqueness" not in dataset.get_field_schema():
        fob.compute_uniqueness(dataset, embeddings="clip_emb")   # 0..1, high = distinctive
    try:
        if "representativeness" not in dataset.get_field_schema():
            fob.compute_representativeness(dataset, embeddings="clip_emb")  # 0..1, high = typical
    except Exception as e:
        print(f"(compute_representativeness unavailable: {e})")

    # keep = passes quality gate AND is not a redundant near-duplicate
    candidates = dataset.match(F("low_quality") == False).match(F("is_duplicate") == False)
    print(f"Candidates after quality gate + dedup: {len(candidates)} / {len(dataset)}")
    shortlist = candidates.sort_by("uniqueness", reverse=True).limit(BUDGET)
    print(f"Shortlist (diversity-ranked): {len(shortlist)} images to annotate")
    return candidates


# ============================================================
# 7. PERSIST SAVED VIEWS  (appear in the App's view dropdown)
# ============================================================
def save_views(dataset, quality_rules, dup_view, uniq_view, has_annotations, has_reference):
    def put(name, view, desc=""):
        if dataset.has_saved_view(name):
            dataset.delete_saved_view(name)
        dataset.save_view(name, view, description=desc)

    # --- A. Quality ---
    put("A1 · Quality — keep for annotation",
        dataset.match(F("low_quality") == False).match(F("is_duplicate") == False)
               .sort_by("uniqueness", reverse=True),
        "Passed the quality gate and not a near-duplicate; most distinctive first.")
    put("A2 · Quality — rejects (low quality)",
        dataset.match(F("low_quality") == True).sort_by("quality"),
        f"Has a rejecting issue ({', '.join(sorted(REJECT_ISSUES))}). "
        "The 'quality' field says which.")

    # one view per issue, worst first. View names carry the threshold, so clear the
    # old ones first (a changed threshold would otherwise leave a stale view behind).
    for name in dataset.list_saved_views():
        if name.startswith(("A3 · Issue", "A4 · Measure")):
            dataset.delete_saved_view(name)
    own = {"corrupt": ("corrupt", "unreadable file"),
           "tiny": ("min_side", f"min_side < {MIN_SIDE}px")}
    for issue in OWN_ISSUES + list(quality_rules):
        if issue in own:
            prop, rule = own[issue]
            lt = True
        else:
            prop, thresh, lt = quality_rules[issue]
            rule = f"{prop} {'<' if lt else '>'} {thresh:g}"
        n = dataset.match(F(issue) == True).count()
        verdict = "rejects" if issue in REJECT_ISSUES else "flag only"
        put(f"A3 · Issue — {issue.replace('_', ' ')} ({rule})",
            dataset.match(F(issue) == True).sort_by(prop, reverse=not lt),
            f"{n} images; {verdict}. Worst first by '{prop}'.")
    # the plugin measures these but defines no issue for them: just rank them
    put("A4 · Measure — most salt & pepper noise",
        dataset.sort_by("salt_and_pepper", reverse=True),
        "Ranked by % isolated black/white pixels (image_issues 'salt_and_pepper').")
    put("A4 · Measure — strongest vignetting",
        dataset.sort_by("vignetting", reverse=True),
        "Ranked by % brightness drop from center to edge (image_issues 'vignetting').")

    # --- B. Duplicates --- (views were captured in audit_duplicates, in order)
    if dup_view is not None:
        put("B1 · Duplicates — near-dup sets",
            dup_view,
            f"Near-duplicates + their representatives at cosine thresh={DUP_THRESH}.")
    if uniq_view is not None:
        put("B2 · Duplicates — unique subset (budget)",
            uniq_view,
            "A maximally-diverse subset of size BUDGET chosen by the index.")

    # --- C. Annotations ---
    if has_annotations:
        put("C1 · Annotations — likely wrong (mistakenness)",
            dataset.sort_by("mistakenness", reverse=True),
            "GT boxes whose label the detector disagrees with, worst first.")
        put("C2 · Annotations — possible missing",
            dataset.match(F("possible_missing") > 0).sort_by("possible_missing", reverse=True),
            "Images where an object looks un-annotated (annotator likely missed it).")
        put("C3 · Annotations — possible spurious",
            dataset.match(F("possible_spurious") > 0).sort_by("possible_spurious", reverse=True),
            "Images with GT boxes that probably shouldn't exist.")
    if has_reference:
        put("C4 · Annotations — planted test errors",
            dataset.match(F("planted") != None).sort_by("mistakenness", reverse=True),
            "The _wrong (every box's class changed) and _duplicate images planted in labels.json.")

    # --- D. Shortlist ---
    put("D1 · Shortlist — annotate (budget, diversity-ranked)",
        dataset.match(F("low_quality") == False).match(F("is_duplicate") == False)
               .sort_by("uniqueness", reverse=True).limit(BUDGET),
        f"The {BUDGET} images worth annotating, most distinctive first.")

    print("\nSaved views:", dataset.list_saved_views())


# ============================================================
# 7b. EXPORT PER-IMAGE SCORES TO CSV
#     (one row per image; every score from every category, if present)
# ============================================================
def export_scores_csv(dataset, gt_field, issue_names=(), path=CSV_PATH):
    schema = dataset.get_field_schema()

    # (csv column, sample field). Only columns whose field actually exists are
    # written, so this stays correct even if a stage was skipped (e.g. no
    # annotations -> no eval_/mistakenness fields; nothing planted -> no planted).
    candidates = [
        ("filepath",           "filepath"),
        ("width",              "metadata.width"),
        ("height",             "metadata.height"),
        # A. quality: image_issues measurements, then one bool per issue
        *[(p, p) for p in QUALITY_PROPS],
        ("min_side",           "min_side"),
        *[(i, i) for i in issue_names],
        ("issues",             "issues"),        # every issue the image has, ";"-joined
        ("n_issues",           "n_issues"),
        ("quality",            "quality"),       # only the issues that reject it
        ("low_quality",        "low_quality"),
        # B. duplicates
        ("is_duplicate",       "is_duplicate"),
        ("nn_distance",        "nn_distance"),
        # curation value
        ("uniqueness",         "uniqueness"),
        ("representativeness", "representativeness"),
        # C. annotation errors
        ("mistakenness",       "mistakenness"),
        ("possible_missing",   "possible_missing"),
        ("possible_spurious",  "possible_spurious"),
        ("eval_tp",            "eval_tp"),
        ("eval_fp",            "eval_fp"),
        ("eval_fn",            "eval_fn"),
        # planted test errors ("wrong" / "duplicate")
        ("planted",            "planted"),
    ]
    cols = [(label, field) for label, field in candidates
            if field.split(".")[0] in schema]

    # one vectorized read per column, all in the same sample order
    data = {label: dataset.values(field) for label, field in cols}
    labels = [label for label, _ in cols]
    if "issues" in data:
        data["issues"] = [";".join(x or []) for x in data["issues"]]

    # detection counts per image (guard: None -> 0)
    n = len(dataset)
    if gt_field in schema:
        gt_counts = dataset.values(F(gt_field + ".detections").length())
        data["n_gt"] = [c or 0 for c in gt_counts]; labels.append("n_gt")
    if PRED_FIELD in schema:
        pr_counts = dataset.values(F(PRED_FIELD + ".detections").length())
        data["n_pred"] = [c or 0 for c in pr_counts]; labels.append("n_pred")

    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(labels)
        for row in zip(*[data[label] for label in labels]):
            w.writerow(["" if v is None else v for v in row])
    print(f"\nWrote per-image scores for {n} images ({len(labels)} columns) -> {path}")


# ============================================================
# 7c. MODEL-EXPLAINABILITY ARTIFACTS
#     (the plain-file contract explainability/failure_analysis.py reads: images.csv,
#     gt.csv, preds.csv, embeddings.npz, points2d.npz, manifest.json)
# ============================================================
def _overlap_xywh(a, b):
    """(IoU, intersection / area of a) matrices for [x, y, w, h] boxes. Both are unchanged by
    scaling x and y independently, so relative FiftyOne boxes give the same values as pixels."""
    if len(a) == 0 or len(b) == 0:
        z = np.zeros((len(a), len(b)))
        return z, z
    iw = np.clip(np.minimum(a[:, None, 0] + a[:, None, 2], b[None, :, 0] + b[None, :, 2])
                 - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, None, 1] + a[:, None, 3], b[None, :, 1] + b[None, :, 3])
                 - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    area_a, area_b = a[:, 2] * a[:, 3], b[:, 2] * b[:, 3]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-12), inter / np.maximum(area_a[:, None], 1e-12)


def _values_or_none(dataset, path):
    """dataset.values(path), or a None per sample if the attribute was never written."""
    try:
        return dataset.values(path)
    except Exception:
        return [None] * len(dataset)


def ensure_explain_fields(dataset, gt_field):
    """MODEL_FIELD and the suspect model's mistakenness on EVERY sample. Loads the
    predictions files and scores them exactly as blackbox_fiftyone.py does."""
    from blackbox_fiftyone import load_predictions   # lazy: blackbox_fiftyone imports this module
    import fiftyone.brain.internal.core.mistakenness as fbm

    schema = dataset.get_field_schema()
    if MODEL_FIELD not in schema or dataset.exists(MODEL_FIELD).count() < len(dataset):
        load_predictions(dataset, MODEL_PREDICTIONS, MODEL_FIELD, 0.01)
    spur = f"possible_spurious_{SUSPECT_TAG}"      # a per-sample count, written for every sample scored
    if spur in schema and dataset.exists(spur).count() == len(dataset):
        print(f"Reusing mistakenness_{SUSPECT_TAG} (suspect-label model)")
        return
    pred = f"pred_{SUSPECT_TAG}"
    load_predictions(dataset, SUSPECT_PREDICTIONS, pred, 0.01)
    fbm._MISSED_CONFIDENCE_THRESHOLD = 0.8          # blackbox_fiftyone's --missing-conf default
    print(f"compute_mistakenness({pred} vs {gt_field}) on all {len(dataset)} samples...")
    fob.compute_mistakenness(dataset, pred, label_field=gt_field,
                             mistakenness_field=f"mistakenness_{SUSPECT_TAG}",
                             missing_field=f"possible_missing_{SUSPECT_TAG}", spurious_field=spur)


def _suspect_flags(gt, sus_boxes, W, H):
    """Per GT box of one image: (suspect_reason, best IoU of any suspect-model box)."""
    gb = np.array([g["box"] for g in gt], float).reshape(-1, 4)
    iou, _ = _overlap_xywh(gb, np.array(sus_boxes or [], float).reshape(-1, 4))
    best = iou.max(1) if iou.shape[1] else np.zeros(len(gt))
    _, cover = _overlap_xywh(gb, gb)
    np.fill_diagonal(cover, 0)
    others = np.array([not g["iscrowd"] for g in gt])
    cover = cover[:, others].max(1) if others.any() else np.zeros(len(gt))
    out = []
    for g, b, c in zip(gt, best, cover):
        reason = ""
        if not g["iscrowd"]:
            if (g["mistakenness"] or 0) >= SUSPECT_CLASS_MIN:
                reason = "class_disagree"
            elif (g["spurious"] and b < SPURIOUS_MAX_IOU and c < SPURIOUS_MAX_COVER
                  and g["box"][2] * W * g["box"][3] * H >= SPURIOUS_MIN_AREA):
                reason = "spurious"
        out.append((reason, float(b)))
    return out


def _print_suspect_calibration(gt_rows):
    """How the suspect thresholds behave. gt_rows: (box, is_original_image). Planted _wrong
    boxes are the answer key for class disagreement; on original COCO images (mostly correct
    labels) the flag rate is a proxy for false flags. There is no planted key for spurious
    boxes, so that part is a sanity read."""
    planted = [g for g, _ in gt_rows if g["planted"] and not g["iscrowd"]]
    orig = [g for g, o in gt_rows if o and not g["iscrowd"]]
    print(f"Suspect-label calibration ({SUSPECT_TAG}; {len(planted)} planted wrong-class boxes, "
          f"{len(orig)} boxes on original images):")
    print(f"  {'class_disagree: mistakenness >=':34s} {'planted caught':>15s} {'originals flagged':>18s}")
    for t in (0.6, 0.7, 0.8, 0.9):
        rec = sum((g["mistakenness"] or 0) >= t for g in planted) / max(len(planted), 1)
        fl = sum((g["mistakenness"] or 0) >= t for g in orig)
        mark = "  <- SUSPECT_CLASS_MIN" if abs(t - SUSPECT_CLASS_MIN) < 1e-9 else ""
        print(f"  {t:<34.2f} {rec:15.0%} {fl:9d} ({fl / max(len(orig), 1):5.1%}){mark}")
    raw = sum(bool(g["spurious"]) for g in orig)
    tight = [g for g in orig if g["reason"] == "spurious"]
    print(f"  spurious: FiftyOne possible_spurious on {raw} original boxes ({raw / max(len(orig), 1):.1%}); "
          f"after the confidence rules {len(tight)} ({len(tight) / max(len(orig), 1):.2%})")
    if tight:
        top = sorted(((sum(g['label'] == c for g in tight), c) for c in {g["label"] for g in tight}), reverse=True)[:6]
        print("    most flagged classes: " + ", ".join(f"{c} ({k})" for k, c in top))


def export_explain_artifacts(dataset, out_dir=EXPLAIN_DIR, include_planted=EXPLAIN_INCLUDE_PLANTED,
                             evaluate_planted=False):
    """evaluate_planted=False (default): file names mean nothing; every image is exported and
    nothing is marked planted. True (test sets): planted copies are recognised, excluded unless
    include_planted, and used as the answer key in the calibration printout."""
    print("\n=== MODEL-EXPLAINABILITY EXPORT ===")
    fields = gt_fields(dataset)
    if not fields:
        print("No ground-truth field; skipping the explainability export.")
        return None
    ensure_explain_fields(dataset, fields[0])
    schema = dataset.get_field_schema()
    m_attr, s_attr = f"mistakenness_{SUSPECT_TAG}", f"possible_spurious_{SUSPECT_TAG}"

    ids, fps, Ws, Hs = dataset.values(["id", "filepath", "metadata.width", "metadata.height"])
    if evaluate_planted:
        planted_img = dataset.values("planted") if "planted" in schema else [None] * len(ids)
        is_missing_copy = [os.path.splitext(os.path.basename(fp))[0].endswith("_missing") for fp in fps]
    else:  # ignore any `planted` field left by an earlier test run
        planted_img, is_missing_copy = [None] * len(ids), [False] * len(ids)
    # GT boxes: per sample, from whichever GT field holds them
    gt_vals = {f: [_values_or_none(dataset, f"{f}.detections.{a}") for a in
                   ("id", "label", "bounding_box", "iscrowd", "planted", m_attr)] for f in fields}
    sus_spur = {f: _values_or_none(dataset, f"{f}.detections.{s_attr}") for f in fields}
    sus_boxes = _values_or_none(dataset, f"pred_{SUSPECT_TAG}.detections.bounding_box")

    gt_rows = []          # (row, is_original_image) for every sample, exported or not
    for i, sid in enumerate(ids):
        f = next((f for f in fields if gt_vals[f][1][i]), None)
        if f is None:
            continue
        did, lab, box, crowd, pl, mis = (v[i] or [] for v in gt_vals[f])
        spur = sus_spur[f][i] or []
        gt = [dict(image_id=sid, gt_idx=k, det_id=did[k], label=lab[k], box=box[k],
                   iscrowd=bool(crowd[k]) if k < len(crowd) else False,
                   planted=evaluate_planted and k < len(pl) and bool(pl[k]),
                   mistakenness=mis[k] if k < len(mis) else None,
                   spurious=bool(spur[k]) if k < len(spur) else False) for k in range(len(lab))]
        for g, (reason, best) in zip(gt, _suspect_flags(gt, sus_boxes[i], Ws[i], Hs[i])):
            g["reason"], g["best_iou"] = reason, best
            gt_rows.append((g, planted_img[i] is None and not is_missing_copy[i]))
    if evaluate_planted:
        _print_suspect_calibration(gt_rows)

    keep = [i for i in range(len(ids)) if include_planted or (planted_img[i] is None and not is_missing_copy[i])]
    keep_ids = {ids[i] for i in keep}
    os.makedirs(out_dir, exist_ok=True)

    def write(name, header, rows):
        with open(os.path.join(out_dir, name), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows([["" if v is None else v for v in r] for r in rows])

    # images.csv
    img_cols = [c for c in QUALITY_PROPS + ["min_side", "nn_distance", "uniqueness", "low_quality",
                                            "is_duplicate", "issues"]
                + (["planted"] if evaluate_planted else []) if c in schema]
    img_vals = {c: dataset.values(c) for c in img_cols}
    if "issues" in img_vals:
        img_vals["issues"] = [";".join(x or []) for x in img_vals["issues"]]
    write("images.csv", ["image_id", "filepath", "width", "height"] + img_cols,
          [[ids[i], fps[i], Ws[i], Hs[i]] + [img_vals[c][i] for c in img_cols] for i in keep])

    # gt.csv
    gt_out = [g for g, _ in gt_rows if g["image_id"] in keep_ids]
    write("gt.csv", ["image_id", "gt_idx", "det_id", "label", "x", "y", "w", "h", "iscrowd", "suspect",
                     "suspect_reason", "yolo_mistakenness", "yolo_max_iou", "planted"],
          [[g["image_id"], g["gt_idx"], g["det_id"], g["label"], *g["box"], int(g["iscrowd"]),
            int(bool(g["reason"])), g["reason"], g["mistakenness"], round(g["best_iou"], 4), int(g["planted"])]
           for g in gt_out])

    # preds.csv (the model being explained, every score down to its 0.01 floor)
    p_id, p_lab, p_box, p_conf = (_values_or_none(dataset, f"{MODEL_FIELD}.detections.{a}")
                                  for a in ("id", "label", "bounding_box", "confidence"))
    pred_out = [[ids[i], k, p_id[i][k], p_lab[i][k], *p_box[i][k], p_conf[i][k]]
                for i in keep for k in range(len(p_lab[i] or []))]
    write("preds.csv", ["image_id", "pred_idx", "det_id", "label", "x", "y", "w", "h", "confidence"], pred_out)

    # embeddings + the 2D map the Embeddings panel already uses
    emb = dataset.values("clip_emb") if "clip_emb" in schema else []
    e_ids = [ids[i] for i in keep if emb and emb[i] is not None]
    if e_ids:
        np.savez(os.path.join(out_dir, "embeddings.npz"), image_ids=np.array(e_ids),
                 emb=np.array([emb[i] for i in keep if emb[i] is not None], np.float32))
    if "clip_viz" in dataset.list_brain_runs():
        viz = dataset.load_brain_results("clip_viz")
        pts = [(sid, xy) for sid, xy in zip(viz.sample_ids, viz.points) if sid in keep_ids]
        np.savez(os.path.join(out_dir, "points2d.npz"), image_ids=np.array([s for s, _ in pts]),
                 xy=np.array([xy for _, xy in pts], np.float32))

    n_sus = Counter(g["reason"] for g in gt_out if g["reason"])
    manifest = dict(source="fiftyone_data_audit.py", dataset=dataset.name, gt_fields=fields,
                    model_field=MODEL_FIELD, model_predictions=MODEL_PREDICTIONS,
                    suspect_model=SUSPECT_TAG, suspect_rules=dict(
                        class_min_mistakenness=SUSPECT_CLASS_MIN, spurious_max_iou=SPURIOUS_MAX_IOU,
                        spurious_min_area_px=SPURIOUS_MIN_AREA, spurious_max_cover=SPURIOUS_MAX_COVER),
                    include_planted=include_planted, evaluate_planted=evaluate_planted,
                    counts=dict(images=len(keep), gt=len(gt_out), preds=len(pred_out),
                                embeddings=len(e_ids), suspect=dict(n_sus)),
                    exported=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Wrote {len(keep)} images, {len(gt_out)} GT boxes ({dict(n_sus)} suspect), "
          f"{len(pred_out)} '{MODEL_FIELD}' predictions"
          f"{' (planted copies excluded)' if evaluate_planted and not include_planted else ''} -> {out_dir}")
    print(f"Next: python explainability/failure_analysis.py --dir \"{out_dir}\"")
    return out_dir


# ============================================================
# 8. SAVE A WORKSPACE  (panel layout, selectable from the App)
# ============================================================
def save_workspace(dataset):
    try:
        samples = fo.Panel(type="Samples", pinned=True)
        side = [fo.Panel(type="Embeddings", state=dict(brainKey="clip_viz")),
                fo.Panel(type="Histograms")]
        if "eval" in dataset.list_evaluations():
            side.append(fo.Panel(type="ModelEvaluation", state=dict(key="eval")))

        # A Space's children must be all Panels (a tab group) OR all Spaces (a split),
        # never a mix. So wrap each pane in its own Space, then split them horizontally:
        # Samples on the left, an Embeddings/Histograms/Model-Eval tab group on the right.
        workspace = fo.Space(
            children=[
                fo.Space(children=[samples]),
                fo.Space(children=side),
            ],
            orientation="horizontal",
        )
        dataset.save_workspace(
            "Data Audit", workspace,
            description="Samples + Embeddings + Histograms + Model-Evaluation",
            overwrite=True,
        )
        print("Saved workspace 'Data Audit'")
    except Exception as e:
        print(f"(workspace save skipped: {e})")


# ============================================================
# MAIN
# ============================================================
def main(explain_dir=EXPLAIN_DIR, include_planted=EXPLAIN_INCLUDE_PLANTED, evaluate_planted=False,
         launch=False):
    dataset = load_dataset()
    compute_embeddings(dataset)

    gt_field = resolve_gt_field(dataset)
    quality_rules = audit_quality(dataset)
    sim, dup_ids, dup_view, uniq_view = audit_duplicates(dataset)
    has_annotations = audit_annotations(dataset, gt_field)
    has_reference = mark_planted(dataset, gt_field) if evaluate_planted else False
    build_shortlist(dataset, dup_ids)

    save_views(dataset, quality_rules, dup_view, uniq_view, has_annotations, has_reference)
    save_workspace(dataset)
    # dump all per-image scores BEFORE launch
    export_scores_csv(dataset, gt_field, issue_names=OWN_ISSUES + list(quality_rules), path=CSV_PATH)
    export_explain_artifacts(dataset, explain_dir, include_planted, evaluate_planted)

    if not launch:
        print(f"\nApp not opened (pass --launch to browse dataset '{DATASET_NAME}'; "
              "the saved views and workspace are already stored)")
        return

    # launch ONCE, opening directly on the quality "keep" view (passing view= at launch
    # avoids the post-launch state_update race that would crash the process, tear down
    # the app server + DB, and leave the browser showing a connection error).
    initial_view = None
    try:
        initial_view = dataset.load_saved_view("A1 · Quality — keep for annotation")
    except Exception as e:
        print(f"(could not load initial view, showing full dataset: {e})")
    session = fo.launch_app(dataset, view=initial_view, port=APP_PORT)
    print(f"\nApp running at http://localhost:{APP_PORT}/datasets/{DATASET_NAME}")
    print("Pick a workflow from the view dropdown (A/B/C/D) and load the 'Data Audit' workspace.")
    session.wait()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="FiftyOne data audit (+ model-explainability export)")
    ap.add_argument("--export-explain", action="store_true",
                    help="only write the explainability artifacts from the existing dataset (no audit, no App)")
    ap.add_argument("--evaluate-planted", action="store_true",
                    help="test sets only: recognise the planted _wrong/_duplicate/_missing copies, report how "
                         "many the audit catches and leave them out of the export (default: off, file names "
                         "mean nothing)")
    ap.add_argument("--include-planted", action="store_true", default=EXPLAIN_INCLUDE_PLANTED,
                    help="with --evaluate-planted: keep the planted copies in the export anyway")
    ap.add_argument("--explain-dir", default=EXPLAIN_DIR)
    ap.add_argument("--launch", action=argparse.BooleanOptionalAction, default=False,
                    help="open the FiftyOne App when the audit finishes (default: don't)")
    ap.add_argument("--csv-path", default=CSV_PATH, help="per-image score table")
    ap.add_argument("--model-predictions", default=MODEL_PREDICTIONS,
                    help="predictions_*.json of the model being explained")
    ap.add_argument("--suspect-predictions", default=SUSPECT_PREDICTIONS,
                    help="predictions_*.json of the independent model that flags suspect labels")
    ap.add_argument("--suspect-tag", default=SUSPECT_TAG,
                    help="name of the suspect model (used in field names: pred_<tag>, mistakenness_<tag>)")
    ap.add_argument("--dataset-name", default=DATASET_NAME,
                    help="FiftyOne dataset to create or reuse; use a new name for a different dataset")
    ap.add_argument("--data-path", default=DATA_PATH, help="folder of images")
    ap.add_argument("--labels-path", default=LABELS_PATH, help="COCO labels to audit")
    args = ap.parse_args()
    # Functions read these module globals; blackbox_fiftyone imports the defaults unchanged.
    CSV_PATH, MODEL_PREDICTIONS, SUSPECT_PREDICTIONS = args.csv_path, args.model_predictions, args.suspect_predictions
    SUSPECT_TAG, DATASET_NAME = args.suspect_tag, args.dataset_name
    DATA_PATH, LABELS_PATH = args.data_path, args.labels_path
    if args.export_explain:
        export_explain_artifacts(load_dataset(), args.explain_dir, args.include_planted, args.evaluate_planted)
    else:
        main(args.explain_dir, args.include_planted, args.evaluate_planted, args.launch)

