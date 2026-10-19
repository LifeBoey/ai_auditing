# Annotation audit: from models + data to a review page

**What this does:** finds the images in a COCO labels file whose *annotations* are most likely
wrong, and builds an HTML page to review them one by one. Every image's annotations are compared
with what your detectors predict. Where they disagree confidently, the image is suspicious.

It checks annotation quality: wrong classes, missing boxes, extra boxes, misplaced boxes. It does
not check image quality (blur, duplicates); `cv_explainability/fiftyone_data_audit.py` does that.

**It needs nothing but your labels, your images and your models.** Nothing is assumed about the
data, file names mean nothing, and there is no answer key. The output is a ranked list of images
to check by hand, not a verdict.

```
trained models + images + labels.json
        │  predict.py
        ▼
predictions_<data>_<model>.json      (one per model)
        │  run_audit.py               3 methods x each model, then an ensemble
        ▼
audit_<labels>/ensemble/review_list.csv    (the most suspect images, ranked)
        │  coco_review_slideshow.py
        ▼
review.html                          (images with annotations + predictions drawn, and why each was flagged)
```

All commands run from this folder (`cv_annotation/`) with the project's Python environment.

---

## 0. Before you start

**Labels:** a COCO detection file with:
- `images`: each has `id`, `file_name`, `width` and `height`
- `annotations`: each has `image_id`, `category_id` and `bbox` as `[x, y, w, h]` in pixels
- `categories`: each has `id` and `name`

**Images:** one folder, with file names exactly as in the labels file.

**Models:** detectors that predict **the same classes as your labels**. A COCO-pretrained detector
can't check blood cells. A box of a class the models don't know can't be checked, and only
produces "unsupported" flags.

**Use models that were not trained on these exact labels, if you can.** A detector trained on an
image learns that image's annotations, mistakes included, and then agrees with them. It
under-reports exactly the errors you're looking for. In order of preference:
1. Models trained on **other** data with the same classes.
2. **Cross-validation:** split the images into K parts (folds). For each fold, train a model on
   the other parts and predict that fold. Merge the K prediction files into one, so every image is
   predicted by a model that never saw it.
3. Models trained on these labels, used as they are. This still finds errors the model didn't
   memorise, such as confident objects nobody annotated. Treat a quiet result as "nothing obvious",
   not "clean".

**More than one model helps.** Each model is one independent opinion. The ensemble trusts an image
more when several models point at it. Two or three models with different architectures (e.g. a
YOLO, a Faster R-CNN) are a good start. A much weaker model mostly adds noise, so check each one's
mAP in step 1 first.

---

## 1. Predictions: `predict.py`

One predictions file per model, over every image in the labels file:

```bash
python predict.py \
    --images  <images_dir> \
    --labels  <labels.json> \
    --model   ultralytics <path/to/yolo/best.pt>            name=yolo \
    --model   fasterrcnn_resnet50_fpn <path/to/frcnn.pt>    name=frcnn \
    --out-dir <predictions_dir> --prefix <data>
```

- **Output:** `<predictions_dir>/predictions_<data>_<name>.json` for each model.
- **Always pass `--labels`.** Image ids and class names then come from your labels, so the
  predictions line up with them. YOLO classes are matched to your categories by name.
- **Keep the default `--conf 0.001`.** The audit needs the low-score boxes, for example to tell
  "the model saw nothing here" from "the model was unsure".
- **The mAP printout** is each model's score against the labels you're auditing. A model with low
  mAP here is either weak or looking at badly labelled data. Compare models with each other rather
  than reading the number on its own.
- **`ARCH`** is `ultralytics` for any YOLO / RT-DETR checkpoint, or a torchvision detector name
  (`fasterrcnn_resnet50_fpn`, `retinanet_resnet50_fpn`, `fcos_resnet50_fpn`,
  `ssdlite320_mobilenet_v3_large`, ...). See `python predict.py --help`.
- **Model options:** add `size=<px>` if the model was trained at a non-default input size. Add
  `num_classes=<n>` if the checkpoint's class count isn't the labels' max category id + 1.

---

## 2. The audit: `run_audit.py`

```bash
python -u run_audit.py \
    --labels      <labels.json> \
    --images-dir  <images_dir> \
    --predictions <predictions_dir>/predictions_<data>_yolo.json,<predictions_dir>/predictions_<data>_frcnn.json \
    --root        audit_<data> \
    2>&1 | tee audit_<data>.log
```

- **`--root`:** the output folder. The default is `audit_<labels file name>`.
- **The FiftyOne dataset is named after `--root`** (here `audit_<data>`). The FiftyOne method keeps
  a permanent dataset, matches images by file name and rewrites its labels to match `--labels`.
  One dataset per audit folder keeps image collections from mixing. A rerun into the same folder
  reuses it. `--fiftyone-dataset <name>` overrides the name.
- **`--budget N`:** how many images go into `review_list.csv`. The default is 10% of the images,
  at least 10.
- **`--reuse`:** skips methods whose output already exists, e.g. after adding a model or after a
  crash.
- **Running time:** cleanlab and FiftyOne are the slow steps. Expect minutes for a few hundred
  images and much longer for thousands.

**What it runs:** three methods ("black boxes") for each predictions file. Each method scores every
image (higher = more suspicious), says yes/no, and gives a one-line reason.

| method | looks for | example reason |
|---|---|---|
| **flags** (`flag_annotations.py`) | rule-based: `wrong_class`, `missing`, `bad_location`, `unsupported`, `duplicate_box`, `invalid_box` | `wrong_class: annotated Platelets; model says WBC 1.00 at IoU 0.96` |
| **cleanlab** (ObjectLab) | per-box scores: swap (wrong class), overlooked (missing), badloc (bad box) | `overlooked: unannotated RBC (0.96) (+3 more)` |
| **fiftyone** (FiftyOne Brain) | mistakenness (wrong class), possible missing, possible spurious | `annotated Platelets: mistakenness 0.99; possible missing: RBC 0.99` |

Then the **ensemble** (`ensemble_audit.py`, run automatically) combines them into one ranking:
1. Every score becomes a percentile, so scales don't matter.
2. The methods built on the same model's predictions form a group, which gets **one vote**. They
   share that model's blind spots.
3. The groups are merged with reciprocal rank fusion. It rewards images that rank near the top for
   several models.

**You don't need to run `ensemble_audit.py` yourself;** `run_audit.py` already did. Rerun it only
to try other weights or a different budget. Give it the same `--images-dir`, and its own
`--out-dir` so the first result isn't overwritten:
```bash
python ensemble_audit.py audit_<data>/scores/* --labels <labels.json> --images-dir <images_dir> \
    --budget 100 --weights "cleanlab=0.5" --out-dir audit_<data>/ensemble_b100
```

---

## 3. The review page: `coco_review_slideshow.py`

```bash
python coco_review_slideshow.py \
    --csv          audit_<data>/ensemble/review_list.csv \
    --labels       <labels.json> \
    --predictions  <predictions_dir>/predictions_<data>_<best model>.json \
    --score-threshold 0.5 \
    --output       review_<data>.html
```

- **Output:** one self-contained HTML file with the images embedded, so it opens offline and can
  be shared. For large lists, add `--copy-images-to <folder>` to save the images next to the page
  instead of embedding them.
- **`--predictions`:** one model's boxes are drawn on each image. Pick your best model.
- **`--score-threshold 0.5`:** draws only confident predictions. The predictions files go down to
  0.001, so without it every image is buried under weak boxes. This only changes what's drawn.
- **Review order:** images appear in `review_list.csv` order, most suspect first.
- **On the page:**
  - Ground truth is green and predictions are red.
  - Toggles: `P` for predictions, `G` for ground truth.
  - Navigation: arrow keys move between images; `+` and `−` zoom.
  - The table under the image gives each method's score and reason. Use the reasons to see what
    to look for.

---

## 4. What you get

| file | what it is |
|---|---|
| `review_<data>.html` | **the review page** |
| `audit_<data>/ensemble/review_list.csv` | **the ranked short list**: `rank`, `file_path`, `final` (ensemble score, 0–1), `n_flagged` / `flagged_by` (which methods said yes), then each method's score and `_reason` |
| `audit_<data>/ensemble/all_ranked.csv` | the same for every image: go further down if the short list keeps finding errors |
| `audit_<data>/ensemble/method_correlation.csv` | how much the methods agree (1 = they rank the images the same way) |
| `audit_<data>/ensemble_nogroups/` | the same ranking with every method voting separately instead of one vote per model |
| `audit_<data>/work/flags_<tag>/flagged_boxes.csv` | **which box to fix**: one row per flag, with the annotated box and class, the model's box, class and score, and the IoU |
| `audit_<data>/work/cleanlab_<tag>_ann/box_issues.csv` | the same from cleanlab, with its per-box issue scores |
| `audit_<data>/scores/<method>_<tag>/scores.csv` | each method's own per-image scores, `flagged` verdict and reason |

`<tag>` is the predictions file name without `predictions_`, e.g. `<data>_yolo`.

---

## 5. Reviewing, and judging the result without an answer key

There's no answer key on real data, so the review itself is the measurement.

1. **Go down the list from the top and keep a tally.** For each image, decide whether there's a
   real annotation error: yes, no, or unsure.
   - The share of "yes" in the first 20–50 images is the precision you can expect from this list.
   - When a stretch of 20 or so images in a row turns up nothing, stop, or switch to
     `all_ranked.csv` if errors were still frequent.
2. **Fix from the box lists, not from memory.** `flagged_boxes.csv` says which box, what's wrong
   with it and what the model saw there.
3. **Trust agreement.** An image flagged by several methods *and* several models (`n_flagged`,
   `flagged_by`) is much more likely real than one flagged by a single method.
4. **Know what each reason usually means:**
   - **`wrong_class` / swap / high mistakenness:** the box is right, the class is wrong. These
     tend to be the most reliable flags.
   - **`missing` / overlooked / possible missing:** the model confidently sees an object with no
     box. It's often a real missing annotation, but check your labelling rules. If the guidelines
     say not to box partly visible or out-of-focus objects, these are expected.
   - **`unsupported` / possible spurious:** an annotated box where the model sees nothing. It may
     be a box that shouldn't exist, or an object the model is bad at, often a small or rare class.
   - **`bad_location` / badloc:** the box is loose, shifted or cut off.
5. **Some things the audit can't see.** Errors on objects the models don't detect (very small,
   rare, heavily occluded, unknown classes) won't surface. A short or quiet list means "the models
   found nothing obvious", not "the labels are clean".
6. **"False alarms" are information too.** If the top of the list is mostly model mistakes on one
   class or one kind of image, the models are weak there. Better or retrained models make every
   method better.

**Tuning, if the list is too noisy or too thin:**
- `--weights` (ensemble): e.g. `cleanlab=0.5` (the default; cleanlab is the noisiest at ranking
  images). Drop a weak model's group with `<tag>=0`, or a single signal with
  `fiftyone.s_possible_spurious=0`. Weight 0 removes it.
- `--combine max` puts anything that any model finds alarming at the top. Use it for a
  high-recall pass.
- Each method's thresholds can be passed through `blackbox_predictions.py` / `blackbox_cleanlab.py`
  (unknown options go to `flag_annotations.py` / `cleanlab_data_audit.py`; see their `--help`).

---

## 6. Worked example (BCCD, three fine-tuned detectors)

```bash
# 1. predictions from the three models trained by DATA/train_bccd_detectors.py
python predict.py --images ../DATA/bccd_images --labels ../DATA/bccd_ground_truth.json \
    --model ultralytics ../DATA/bccd_training_runs/yolo11n/weights/best.pt \
    --model frcnn_mbv3  ../DATA/bccd_training_runs/frcnn_mbv3/best.pt \
    --model ssdlite     ../DATA/bccd_training_runs/ssdlite/best.pt size=512 \
    --out-dir ../DATA/prediction_jsons --prefix bccd

# 2. the audit (FiftyOne dataset: audit_bccd, after --root)
python -u run_audit.py --labels ../DATA/bccd_ground_truth.json --images-dir ../DATA/bccd_images \
    --predictions ../DATA/prediction_jsons/predictions_bccd_frcnn_mbv3.json,../DATA/prediction_jsons/predictions_bccd_yolo11n.json,../DATA/prediction_jsons/predictions_bccd_ssdlite.json \
    --root audit_bccd 2>&1 | tee audit_bccd.log

# 3. the review page, with the strongest model's boxes drawn
python coco_review_slideshow.py --csv audit_bccd/ensemble/review_list.csv \
    --labels ../DATA/bccd_ground_truth.json \
    --predictions ../DATA/prediction_jsons/predictions_bccd_frcnn_mbv3.json \
    --score-threshold 0.5 --output review_bccd.html
```

Things to keep in mind on this example:
- **Training images:** these detectors were trained on 80% of these same images, so see
  "Use models that were not trained on these exact labels" in section 0.
- **SSDLite and platelets:** SSDLite detects no platelets at all (AP 0), so its group mostly adds
  noise on that class. Try `--weights "cleanlab=0.5,bccd_ssdlite=0"` in a re-ensemble and compare
  the two lists.

---

## 7. Gotchas

- **One FiftyOne dataset per audit folder.** It's named after `--root` unless you pass
  `--fiftyone-dataset`. The method imports any images that are new to that dataset and syncs its
  class labels to your `--labels`, so running it changes that dataset. Don't point two image
  collections whose file names overlap at the same `--root` (or the same `--fiftyone-dataset`).
  Old datasets stay in FiftyOne until deleted: `fo.delete_dataset("<name>")`.
- **Classes must match between models and labels.** `predict.py --labels` takes care of this. With
  prediction files from elsewhere, `class_id` must be your labels' `category_id`, and `class_name`
  your category name.
- **The predictions format** these scripts read is one entry per image:
  `[{"file_name", "prediction": [{"class_id", "class_name", "score", "bbox": [x1, y1, x2, y2]}]}]`.
  `predict.py` writes it by default.
- **`--evaluate-planted` and `analyze_audit_artifacts.py` are for test sets only.** They measure
  the audit itself on deliberately corrupted copies (`<name>_wrong.jpg`, `<name>_missing.jpg`,
  plus a log of what was changed). On real data, leave both out. They have nothing to measure
  against, and file names carry no meaning.
