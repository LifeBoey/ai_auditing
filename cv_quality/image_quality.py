"""Score every image in IMAGE_DIR with three tools and write one CSV per tool:

  image_quality_checker.csv  -> image_quality_score + the factors behind it + analyze_blur
  cleanvision.csv            -> Imagelab per-issue scores/flags + an overall score
  fiftyone.csv               -> image_issues plugin measurements/flags + an overall score

then combines them into ensemble.csv and saves annotated figures of the TOP_K worst
(or best) images to INSPECTION_DIR.
"""
import contextlib
import importlib.util
import io
import os
import textwrap
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")       # never open windows: figures are only saved to disk
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from scipy.stats import entropy

# ---------------------------------------------------------------- configuration
IMAGE_DIR = "/mnt/c/Users/Gabriel/fiftyone/coco-2017/validation/data"     # folder of images (searched recursively)
OUTPUT_DIR = "csvs"                           # where the 3 CSVs are written
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IQC_THRESHOLD = 0.5                        # image_quality_score below this = low quality
RESCORE = True             # False = reuse the 3 CSVs in OUTPUT_DIR if they already exist
TOP_K = 25                 # how many images to annotate
TOP_K_WORST = True         # True = the K worst images, False = the K best
INSPECTION_DIR = "for_inspection"


def list_images(image_dir):
    return sorted(os.path.join(root, f)
                  for root, _, files in os.walk(image_dir) for f in files
                  if os.path.splitext(f)[1].lower() in IMAGE_EXTS)


# =================================================== 1) image_quality_checker
from image_quality_checker import image_quality_score, analyze_blur
from image_quality_checker import quality as iqc_quality
from image_quality_checker.utils import crop_foreground

# Same weights / normalisers image_quality_score uses internally; it only returns
# the final number, so the per-factor breakdown is recomputed here the same way.
IQC_WEIGHTS = {"blur_score": 0.1, "resolution": 0.1, "brightness": 0.1, "contrast": 0.1,
               "motion_blur": 0.1, "edge_density": 0.25, "color_std_dev": 0.15,
               "entropy": 0.1}
IQC_MAX = {"resolution": 1_000_000, "brightness": 255, "contrast": 255, "motion_blur": 100,
           "edge_density": 0.05, "color_std_dev": 100, "entropy": 8.0}


def _iqc_factors(image_bytes):
    """Mirror of image_quality_score's internals, returning every factor."""
    out = {"file_size_kb": len(image_bytes) / 1024}
    if len(image_bytes) < 3 * 1024:
        return out
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return out
    out["width"], out["height"] = img.shape[1], img.shape[0]

    img = cv2.resize(crop_foreground(img), (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    from sklearn.cluster import KMeans
    pixels = img.reshape(-1, 3)
    sample = pixels[np.random.choice(pixels.shape[0], 1000, replace=False)]
    labels = KMeans(n_clusters=5, n_init=10).fit(sample).labels_
    out["dominant_color_pct"] = max(np.bincount(labels)) / len(sample) * 100
    if out["dominant_color_pct"] > 93:
        return out

    faces = iqc_quality.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    out["n_faces"] = len(faces)
    out["is_portrait"] = len(faces) > 0
    blur_threshold = 30 if out["is_portrait"] else 60
    out["blur_threshold"] = blur_threshold

    center = gray[gray.shape[0] // 4: 3 * gray.shape[0] // 4,
                  gray.shape[1] // 4: 3 * gray.shape[1] // 4]
    # The library takes log(0) = -inf on exact-zero frequency bins (flat regions such
    # as blown-out sky or black bars), which makes motion_blur -- and its whole score --
    # NaN. Skip those bins; for every other image this is identical to the library.
    mag = np.abs(np.fft.fftshift(np.fft.fft2(center)))
    spectrum = 20 * np.log(mag[mag > 0])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist, _ = np.histogram(gray.ravel(), bins=256, range=[0, 256])
    f = {"blur_score": cv2.Laplacian(gray, cv2.CV_64F).var(),
         "resolution": img.shape[0] * img.shape[1],     # after the 256x256 resize
         "brightness": np.mean(gray),
         "contrast": np.std(gray),
         "motion_blur": np.std(spectrum),
         "edge_density": np.sum(cv2.Canny(gray, 100, 200) > 0) / gray.size,
         "color_std_dev": np.mean([np.std(hsv[:, :, i]) for i in range(3)]),
         "entropy": entropy(hist, base=2)}
    norm = dict(IQC_MAX, blur_score=blur_threshold)
    for k, v in f.items():
        out[k] = float(v)
        out[f"{k}_contribution"] = IQC_WEIGHTS[k] * min(v / norm[k], 1.0)
    return out


def _iqc_one(path):
    row = {"filepath": path, "filename": os.path.basename(path)}
    try:
        with open(path, "rb") as fh:
            image_bytes = fh.read()
        # same seed for both calls so the random KMeans colour check agrees
        np.random.seed(0)
        # library prints every score, and warns on the log(0) handled below
        with contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"):
            result = image_quality_score(image_bytes)
        np.random.seed(0)
        factors = _iqc_factors(image_bytes)
    except Exception as e:
        result, factors = e, {}

    # image_quality_score returns: float score | True (tiny / single colour / corrupt)
    #                              | False (undecodable) | an Exception
    if isinstance(result, Exception):
        row.update(iqc_score=np.nan, status=f"error: {result}".strip())
    elif result is True:
        row.update(iqc_score=0.0, status="rejected: tiny, single colour or corrupt")
    elif result is False:
        row.update(iqc_score=0.0, status="rejected: could not decode")
    elif np.isnan(result) and "motion_blur_contribution" in factors:
        # library score is NaN from its log(0) motion_blur bug -- rebuild it from the
        # same weighted factors with motion_blur computed safely
        row.update(iqc_score=sum(factors[f"{k}_contribution"] for k in IQC_WEIGHTS),
                   status="scored (motion_blur fixed)")
    else:
        row.update(iqc_score=float(result), status="scored")
    row["low_quality"] = not (row["iqc_score"] >= IQC_THRESHOLD)
    row.update(factors)

    try:
        blur = analyze_blur(path)
        row.update(blur_ratio_pct=blur["blur_ratio"], blur_category=blur["category"],
                   overall_blur=blur["overall_blur"])
    except Exception as e:
        row.update(blur_ratio_pct=np.nan, blur_category=f"error: {e}", overall_blur=np.nan)
    return row


def run_image_quality_checker(paths, out_csv):
    print(f"\n=== image_quality_checker: {len(paths)} images ===")
    rows = []
    for i, p in enumerate(paths, 1):
        rows.append(_iqc_one(p))
        if i % 50 == 0 or i == len(paths):
            print(f"  {i}/{len(paths)}")
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}  (low quality: {int(df['low_quality'].sum())}/{len(df)})")
    return df


# ======================================================== 2) cleanvision Imagelab
DUPLICATE_ISSUES = {"exact_duplicates", "near_duplicates"}


def _is_readable(path):
    try:
        with Image.open(path) as im:
            im.load()                      # full decode: truncated files raise here
        return True
    except Exception:
        return False


def run_cleanvision(paths, out_csv):
    from cleanvision.imagelab import Imagelab
    print("\n=== cleanvision Imagelab ===")
    # Imagelab crashes outright on a truncated/corrupt file, so give it only readable
    # ones and add the rest back as `corrupt` rows
    readable = [p for p in paths if _is_readable(p)]
    corrupt = [p for p in paths if p not in set(readable)]
    if corrupt:
        print(f"Skipping {len(corrupt)} unreadable image(s) (marked corrupt in the CSV)")
    imagelab = Imagelab(filepaths=readable)
    imagelab.find_issues()
    # imagelab.report() also plots example images; only the summary table is wanted
    print(imagelab.issue_summary.to_string(index=False))

    df = imagelab.issues.copy()        # one row per image, index = filepath
    issue_names = [c[3:-6] for c in df.columns if c.startswith("is_") and c.endswith("_issue")]
    df["is_corrupt_issue"] = False
    if corrupt:
        bad = pd.DataFrame({"is_corrupt_issue": True}, index=corrupt)
        for n in issue_names:
            bad[f"{n}_score"] = 0.0 if n not in DUPLICATE_ISSUES else np.nan
        df = pd.concat([df, bad])
    issue_names.append("corrupt")
    flag_cols = [f"is_{n}_issue" for n in issue_names]
    df[flag_cols] = df[flag_cols].eq(True)          # NaN (from the corrupt rows) -> False
    # each <issue>_score is 0..1 where LOW = issue more severe; duplicates are a
    # property of the set rather than the image, so they are left out of the overall
    score_cols = [f"{n}_score" for n in issue_names
                  if n not in DUPLICATE_ISSUES and f"{n}_score" in df.columns]
    df["overall_quality_score"] = df[score_cols].min(axis=1)     # worst factor
    df["mean_quality_score"] = df[score_cols].mean(axis=1)
    df["issues"] = df.apply(lambda r: ";".join(n for n in issue_names if r[f"is_{n}_issue"]),
                            axis=1)
    df["n_issues"] = df[[f"is_{n}_issue" for n in issue_names]].sum(axis=1)
    df["low_quality"] = df[[f"is_{n}_issue" for n in issue_names
                            if n not in DUPLICATE_ISSUES]].any(axis=1)

    df.index.name = "filepath"
    df = df.reset_index()
    df.insert(1, "filename", df["filepath"].map(os.path.basename))
    front = ["filepath", "filename", "overall_quality_score", "mean_quality_score",
             "low_quality", "n_issues", "issues"]
    df = df[front + [c for c in df.columns if c not in front]]
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}  (images with issues: {int(df['low_quality'].sum())}/{len(df)})")
    return df


# =================================================== 3) fiftyone image_issues plugin
import fiftyone as fo
import fiftyone.core.utils as fou

IMAGE_ISSUES_PLUGIN = os.path.join(
    fo.config.plugins_dir, "@jacobmarks", "image_issues", "__init__.py"
)
# Per-image measurements written by the plugin (exposure splits into min/max)
QUALITY_PROPS = ["aspect_ratio", "blurriness", "brightness", "contrast", "entropy",
                 "min_exposure", "max_exposure", "salt_and_pepper", "saturation",
                 "vignetting"]
# Fields from this script's old home-grown quality check. `brightness`/`contrast`
# reuse the plugin's names on a DIFFERENT scale, and the plugin skips any field that
# already exists, so they must be dropped before the plugin values are written.
LEGACY_QUALITY_FIELDS = ["blur", "brightness", "contrast", "is_corrupt"]
OWN_ISSUES = ["corrupt", "tiny"]   # the only checks the plugin doesn't cover
REJECT_ISSUES = {"corrupt", "tiny", "blurry", "dark", "low_entropy", "low_contrast"}
QUALITY_WORKERS = 8     # threads for the per-image measurements (OpenCV releases the GIL)
ISSUE_THRESHOLDS = {}
MIN_SIDE = 100
ImageFile.LOAD_TRUNCATED_IMAGES = False


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


def run_fiftyone(image_dir, out_csv):
    print("\n=== fiftyone image_issues ===")
    dataset = fo.Dataset.from_images_dir(image_dir, recursive=True)   # temporary dataset
    try:
        dataset.compute_metadata()
        rules = audit_quality(dataset)
        all_issues = OWN_ISSUES + list(rules)
        reject_checks = [n for n in all_issues if n in REJECT_ISSUES]

        cols = (["filepath", "metadata.width", "metadata.height", "min_side"]
                + QUALITY_PROPS + all_issues + ["n_issues", "quality", "low_quality"])
        df = pd.DataFrame({c.split(".")[-1]: dataset.values(c) for c in cols})
        df["issues"] = [";".join(x) for x in dataset.values("issues")]
        # the plugin has no single score: overall = share of rejecting checks passed
        # (1.0 = clean, 0.0 = failed all of them); also the share of every check passed
        n_reject = df[reject_checks].sum(axis=1)
        df["overall_quality_score"] = 1 - n_reject / len(reject_checks)
        df["checks_passed_score"] = 1 - df["n_issues"] / len(all_issues)
        # a corrupt image can't be measured, so it "passes" the other checks -- force 0
        df.loc[df["corrupt"], ["overall_quality_score", "checks_passed_score"]] = 0.0
        df.insert(1, "filename", df["filepath"].map(os.path.basename))

        front = ["filepath", "filename", "overall_quality_score", "checks_passed_score",
                 "low_quality", "quality", "n_issues", "issues"]
        df = df[front + [c for c in df.columns if c not in front]]
        df.to_csv(out_csv, index=False)
        print(f"Wrote {out_csv}")
        return df
    finally:
        dataset.delete()


# ============================================================ 4) ensemble + inspection
TOOLS = ["iqc", "cleanvision", "fiftyone"]


def _key(path):
    """The tools report paths differently (relative/absolute, / vs \\) -- normalise."""
    return os.path.normcase(os.path.abspath(path))


def _issue_list(s):
    return [] if pd.isna(s) or s == "" else str(s).split(";")


def _iqc_summary(r):
    """Short text: blur category + the weakest factors of the score."""
    if not str(r["status"]).startswith("scored"):
        return r["status"]
    weak = []
    for k, w in IQC_WEIGHTS.items():
        c = r.get(f"{k}_contribution")
        if k != "resolution" and pd.notna(c) and c / w < 0.5:   # resolution is constant
            weak.append((c / w, k))
    parts = [str(r["blur_category"])] + [k for _, k in sorted(weak)[:3]]
    return ", ".join(parts)


def build_ensemble(iqc_csv, cv_csv, fo_csv, out_csv):
    """One row per image. Each tool's score is turned into a percentile rank (0 = worst
    image in the set, 1 = best) so the three different scales are comparable, and
    ensemble_score is the mean of those ranks. tools_flagged counts how many of the
    three tools marked the image low quality."""
    iqc, cv, fo_ = (pd.read_csv(p) for p in (iqc_csv, cv_csv, fo_csv))
    tables = {
        "iqc": iqc.assign(score=iqc["iqc_score"], summary=iqc.apply(_iqc_summary, axis=1)),
        "cleanvision": cv.assign(score=cv["overall_quality_score"],
                                 summary=cv["issues"].map(lambda s: ", ".join(_issue_list(s)))),
        "fiftyone": fo_.assign(score=fo_["overall_quality_score"],
                               summary=fo_["issues"].map(lambda s: ", ".join(_issue_list(s)))),
    }
    merged = None
    for tool, t in tables.items():
        t = t.assign(key=t["filepath"].map(_key))[["key", "filepath", "score",
                                                   "low_quality", "summary"]]
        t = t.rename(columns={c: f"{tool}_{c}" for c in t.columns if c != "key"})
        merged = t if merged is None else merged.merge(t, on="key", how="outer")

    df = pd.DataFrame({"filepath": merged[[f"{t}_filepath" for t in TOOLS]].bfill(axis=1)
                       .iloc[:, 0]})
    df["filename"] = df["filepath"].map(os.path.basename)
    for t in TOOLS:
        df[f"{t}_rank"] = merged[f"{t}_score"].rank(pct=True, method="average")
    df["ensemble_score"] = df[[f"{t}_rank" for t in TOOLS]].mean(axis=1)   # skips NaN
    df["tools_flagged"] = sum(merged[f"{t}_low_quality"].fillna(False).astype(bool)
                              for t in TOOLS)
    for t in TOOLS:
        df[f"{t}_score"] = merged[f"{t}_score"]
        df[f"{t}_low_quality"] = merged[f"{t}_low_quality"]
        df[f"{t}_issues"] = merged[f"{t}_summary"].fillna("")
    # worst first: most tools flagging it, then lowest ensemble score
    df = df.sort_values(["tools_flagged", "ensemble_score"], ascending=[False, True])
    df.insert(0, "worst_rank", range(1, len(df) + 1))
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")
    return df


def _load_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is not None:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"))
    except Exception:
        return None


def save_inspection(df, k, out_dir, worst=True):
    os.makedirs(out_dir, exist_ok=True)
    picked = df.head(k) if worst else df.iloc[::-1].head(k)
    label = "worst" if worst else "best"
    for i, r in enumerate(picked.itertuples(index=False), 1):
        img = _load_rgb(r.filepath)
        fig, ax = plt.subplots(figsize=(8, 8))
        if img is None:
            ax.set_facecolor("lightgrey")
            ax.text(0.5, 0.5, "image could not be read", ha="center", va="center",
                    fontsize=14, transform=ax.transAxes)
        else:
            ax.imshow(img)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(f"#{i} {label}: {r.filename}\nensemble score {r.ensemble_score:.2f} "
                     f"| flagged by {r.tools_flagged}/3 tools", fontsize=11)

        lines = []
        for t, name in [("iqc", "image_quality_checker"), ("cleanvision", "CleanVision"),
                        ("fiftyone", "FiftyOne")]:
            score = getattr(r, f"{t}_score")
            issues = getattr(r, f"{t}_issues") or "no issues"
            low = getattr(r, f"{t}_low_quality")
            flag = " [LOW]" if pd.notna(low) and bool(low) else ""
            s = "n/a" if pd.isna(score) else f"{score:.2f}"
            lines.append(textwrap.fill(f"{name} {s}{flag}: {issues}", 95,
                                       subsequent_indent="    "))
        ax.set_xlabel("\n".join(lines), loc="left", fontsize=9, family="monospace",
                      bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

        stem = os.path.splitext(r.filename)[0]
        fig.savefig(os.path.join(out_dir, f"{i:02d}_{label}_{stem}.png"), dpi=110,
                    bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {len(picked)} annotated images to {out_dir}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csvs = {n: os.path.join(OUTPUT_DIR, f"{n}_qualaudit.csv")
            for n in ("image_quality_checker", "cleanvision", "fiftyone")}
    if RESCORE or not all(os.path.isfile(p) for p in csvs.values()):
        paths = list_images(IMAGE_DIR)
        if not paths:
            raise SystemExit(f"No images found in {IMAGE_DIR!r}")
        run_image_quality_checker(paths, csvs["image_quality_checker"])
        run_cleanvision(paths, csvs["cleanvision"])
        run_fiftyone(IMAGE_DIR, csvs["fiftyone"])
    ens = build_ensemble(csvs["image_quality_checker"], csvs["cleanvision"],
                         csvs["fiftyone"], os.path.join(OUTPUT_DIR, "ensemble.csv"))
    save_inspection(ens, TOP_K, INSPECTION_DIR, worst=TOP_K_WORST)
