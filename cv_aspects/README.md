# cv_aspects — how does a detector's performance depend on an aspect of the image?

You give it images, ground truth and a model's predictions. It splits the images into groups by
some **aspect**, such as brightness, number of objects, visual cluster, or anything your own code
decides. Then it scores the model separately on each group, so you can see which aspects change its performance.

It works in two stages, which only share one file (`tags.csv`):

```
 images (+ GT, + CLIP)  ──►  1. split_by_aspect.py  ──►  tags.csv  ──►  2. evaluate_by_aspect.py  ──►  metrics + plots
                                                                            ▲
                                             predictions + GT ──────────────┘
```

Run stage 1 once per dataset. Stage 2 can then be run for as many models as you like.

---

## 1. Requirements

| | |
|---|---|
| **Required** | Python 3.10+, `numpy`, `pandas`, `pillow`, `scikit-learn`, `matplotlib`, `pycocotools` |
| Optional | `torch` + `open_clip_torch`, only for the `clip_cluster` aspect (`--clip`) |

Run the commands from inside `cv_aspects/`. The examples use the BCCD data in `../DATA`.

## 2. What you need to bring

| Input | Stage 1 | Stage 2 | Format |
|---|---|---|---|
| Image folder | **required** (`--images`) | – | `.jpg/.png/...` files |
| Ground truth | optional (`--labels`), enables the box-based aspects | **required** (`--labels`) | COCO json (`images`, `annotations`, `categories`) |
| Predictions | – | **required** (`--preds`) | see below |
| CLIP weights | optional (`--clip`), enables `clip_cluster` | – | OpenAI CLIP `.pt`, e.g. `/mnt/c/Users/Gabriel/fiftyone/__models__/CLIP-ViT-B-32.pt` |
| Config | optional (`--config`) | – | JSON with your binning / plugin settings |
| Your own aspect | optional (`--plugin`) | – | a `.py` file made from `plugin_template.py` |

**Predictions** can be in either of two formats:
- Standard COCO results: `[{"image_id", "category_id", "bbox": [x, y, w, h], "score"}]`
- Per-image lists: `[{"file_name", "prediction": [{"class_name" or "class_id", "bbox": [x1, y1, x2, y2], "score"}]}]`

  Images are matched to the GT by `file_name`, and classes by `class_name` (falling back to `class_id`).

Include low-score boxes too (e.g. down to 0.001). AP needs them, and the tool chooses the score threshold itself.

## 3. Steps

### Step 1: split the images into groups

```
python split_by_aspect.py --images ../DATA/bccd_images \
    --labels ../DATA/bccd_ground_truth.json \
    --out aspects_bccd
    --clip /mnt/c/Users/Gabriel/fiftyone/__models__/CLIP-ViT-B-32.pt
```

- Only `--images` is required. Without `--labels` the box-based aspects are skipped, and without `--clip` `clip_cluster` is skipped. Skipped aspects are listed at the end of the run.
- `--aspects brightness,contrast` runs only some aspects. `--list` shows all of them.
- `--out DIR` sets the output folder. The default is `cv_aspects/aspects_out/`.
- An aspect that has the same value on every image is dropped with a message. On BCCD, for example, `aspect_ratio` is dropped because every image is 640×480.

### Step 2: score a model on every group

```
python evaluate_by_aspect.py --preds ../DATA/prediction_jsons/predictions_bccd_frcnn_mbv3.json \
    --labels ../DATA/bccd_ground_truth.json --tags aspects_out/tags.csv
```

To compare models, repeat this with a different `--preds` each time. Each model gets its own output folder.

| Option | Default | Meaning |
|---|---|---|
| `--conf` | `auto` | Score threshold for precision / recall / F1. `auto` picks the best-F1 threshold on all images, and the same threshold is then used for every group. |
| `--iou` | `0.5` | IoU used for precision / recall (mAP always averages over 0.50–0.95) |
| `--min-group` | `20` | Groups with fewer images are flagged `low_n` and left out of the ranking |
| `--bootstrap` | `200` | Resamples used for the AP50 confidence interval (`0` = off, faster) |
| `--out` | `<tags folder>/<predictions file name>/` | Output folder |

### Optional: verify the numbers

```
python check_against_pycocotools.py --preds ... --labels ... --tags aspects_out/tags.csv
```

This recomputes every group with pycocotools and checks that each aspect's groups add up to the overall totals.

---

## 4. Choosing how groups are made (optional config)

Numeric aspects are split into 4 equal-sized groups (quantiles) by default. Pass `--config my.json` to change this:

```json
{
  "default_n_bins": 4,
  "brightness":   {"edges": [150, 180], "names": ["dark", "normal", "bright"]},
  "n_gt":         {"n_bins": 3},
  "clip_cluster": {"k": 5}
}
```

- `edges` gives fixed thresholds, and each bin includes its left edge: `< 150`, `150–180`, `≥ 180`. `names` is optional.
- `n_bins` sets the number of quantile groups for that aspect.
- Automatic group names look like `2: [179, 181)`: group number, then value range (`[` = included, `)` = excluded).

**Built-in aspects:**

| Needs | Aspects |
|---|---|
| images | `brightness`, `contrast`, `sharpness` (low = blurry), `saturation`, `hue`, `aspect_ratio`, `megapixels` |
| `--labels` | `n_gt`, `n_classes`, `crowding` (overlap between GT boxes), `small_frac`, `mean_box_area`, and for each class `n_<class>` and `has_<class>` |
| `--clip` | `clip_cluster` (images grouped by visual similarity) |

## 5. Bringing your own aspect (plugin)

You don't need to edit this package to add your own aspect:

1. Copy [`plugin_template.py`](plugin_template.py) to anywhere you like.
2. Fill in the three parts marked `>>> FILL IN <<<`:
   1. `NAME` / `KIND` / `DESCRIPTION`
   2. `setup()`, which loads your model once
   3. `value_for_image()`, which returns the value for one image
3. Run it:

```
python split_by_aspect.py --images ../DATA/bccd_images --plugin my_aspect.py --aspects plugins
```

`KIND` decides what the tool does with your values:

| `KIND` | Return per image | Groups |
|---|---|---|
| `categorical` | a label, e.g. `"sunny"` | one per label |
| `numeric` | a number | binned as in section 4 |
| `embedding` | a vector | KMeans clusters (`k` in the config, default 5) |

Return `None` when your code can't decide; those images go into the group `n/a`. Settings for your plugin, such as a model path, go in the config under your plugin's `NAME`. They reach your code as `options`.

See [`plugins/odd_even.py`](plugins/odd_even.py) for a filled-in example, which groups images by whether their file name ends in an odd or even number.

**Skipping stage 1:** you can also write `tags.csv` yourself. It needs a `file_name` column plus one column per aspect called `<aspect>__group`.

---

## 6. Outputs — what to look at

### From stage 1 (`aspects_out/`)

| File | What it is | Use it to |
|---|---|---|
| `tags.csv` | One row per image: `file_name`, `image_id`, and for each aspect the raw value (`<aspect>`) and its group (`<aspect>__group`) | see which group an image fell into; input for stage 2 |
| `tags.json` | For each aspect: how it was binned (edges), group sizes, which aspects were skipped or dropped and why | check the groups are sensible (not one huge group and several tiny ones) |
| `plots/groups_<aspect>.png` | 6 sample thumbnails per group | **see what a group actually looks like**, which is especially useful for `clip_cluster` |
| `clip_emb.npz` | Cached CLIP embeddings | nothing; it just makes reruns fast |

### From stage 2 (`aspects_out/<model>/`)

**Start here:**

1. **`plots/ranking.png`: which aspects matter most.**
   - Each bar is one aspect. Its length is the *AP50 spread*: the best group's AP50 minus the worst group's.
   - The label names the worst group.
   - Lighter bars at the bottom are aspects whose groups are too small to trust (fewer than `--min-group` images).
2. **`plots/<aspect>.png`: how one aspect affects the model.**
   - Three panels: AP50, recall and precision for each group.
   - Error bars are 95% confidence intervals, and the dashed line is the score on all images.
   - Lighter bars are groups smaller than `--min-group`.
   - If the bars stay near the dashed line and the error bars overlap, that aspect doesn't affect the model.
3. **`metrics_by_aspect.csv`: the exact numbers behind the plots**, one row per (aspect, group), plus an `overall` row. The same data is in `metrics_by_aspect.json`.

**Columns in `metrics_by_aspect.csv`:**

| Column | Meaning |
|---|---|
| `aspect`, `group`, `n_images` | which group, and how many images are in it |
| `mAP`, `AP50` | COCO mAP@[.50:.95] and AP at IoU 0.5 on this group's images only; threshold-free |
| `AP50_lo`, `AP50_hi` | 95% bootstrap interval for AP50 |
| `AP50_<class>` | AP50 for each class within the group (empty if the class doesn't occur in it) |
| `score_threshold`, `iou` | the threshold and IoU used for the counts below (the same for every group) |
| `n_gt`, `TP`, `FP`, `FN`, `n_pred` | GT objects, correct detections, false alarms, missed objects, and kept detections |
| `precision`, `recall`, `F1` | at that threshold; `_lo` / `_hi` give 95% Wilson intervals |
| `d_mAP`, `d_AP50`, `d_recall`, `d_precision`, `d_F1` | **group minus overall**: negative means the model does worse on this group |
| `recall_p`, `precision_p` | p-value that this group's rate differs from **the rest of the images** |
| `recall_q`, `precision_q` | the same p-value corrected for testing many groups at once; **below 0.05 means the difference is probably real** |
| `low_n` | `True` if the group has fewer than `--min-group` images; treat its numbers with caution |

**Columns in `aspect_ranking.csv`** (the table behind `ranking.png`):

| Column | Meaning |
|---|---|
| `trusted` | `False` if fewer than 2 groups are big enough; such aspects are sorted last |
| `AP50_spread`, `recall_spread` | best group minus worst group |
| `worst_group`, `worst_AP50`, `best_group` | which groups those are |
| `worst_d_recall`, `worst_recall_q` | how the worst group's recall compares with the rest, and whether that difference is real |

**Example: "does brightness affect my model?"**
1. Open `plots/brightness.png`.
2. Look up the `brightness` rows in `metrics_by_aspect.csv`: check `d_AP50` / `d_recall` and whether `recall_q` is below 0.05.
3. Find where `brightness` sits in `ranking.png`.

## 7. Reading the results carefully

- **Spreads below about 0.03 AP50 are probably noise.** A group of ~90 images has an AP50 uncertainty of about ±0.015 on its own. Check whether the error bars overlap.
- **The ranking uses AP50 only.** Recall can behave differently; for example, on BCCD `n_gt` has a small AP50 spread but the largest recall spread. Look at the per-aspect plots too.
- **Aspects with more groups get bigger spreads by chance**: 5 clusters versus 2 groups, for instance.
- **The `_q` values compare each group with the rest of the images.** One very bad group makes all the other groups look significantly better.
- **A difference shows a link, not a cause.** For example, a brightness group may also differ in how many objects it contains.
- **Only the values present in your data are tested.** If none of your images are dark, the results say nothing about dark images.
