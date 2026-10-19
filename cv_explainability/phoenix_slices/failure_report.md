# Model failure report

Generated 2026-10-13 10:19 from `C:\Users\Gabriel\fiftyone\coco-2017\validation\explainability\phoenix_slices` (gt_units.csv, pred_units.csv).

## 1. Overview (what Phoenix shows)

Each table is one kind of unit, and each row of it one span in Phoenix. The `correct` annotation summary in Phoenix is the accuracy column below; filter spans and it updates for that slice.

| unit | rows | correct | failures |
|---|---|---|---|
| `gt_object` (gt_units.csv) | 37,754 | 65.2% | missed 53%, mislocalized 35%, confused 11%, claimed_by_neighbour 2% |
| `prediction` (pred_units.csv) | 42,277 | 58.3% | mislocalized 46%, background 39%, wrong_class 10%, duplicate 5% |

- **gt_object**: 65.2% of 37,754 objects are correct; the commonest failure is *missed* (53% of failures), then *mislocalized* (35%). Biggest problem area: `size = small` (55.3% not found vs 25.3%).
- **prediction**: 58.3% of 42,277 predictions are correct; the commonest failure is *mislocalized* (46% of failures), then *background* (39%). Biggest problem area: `score (quintile) = 0.5 to 0.642` (79.7% wrong vs 32.2%).

## 2. Main problems

One entry per attribute, ranked by its excess failures (failures beyond what the overall rate predicts, summed over all its significantly worse values) as a share of that unit's failures. Slices that another attribute fully explains are in section 3 instead.

1. **prediction: score (quintile) = 0.5 to 0.642** (and 1 more values): 79.7% vs 32.2%, 37% of prediction failures are excess
2. **gt_object: size = small**: 55.3% vs 25.3%, 27% of object failures are excess, mostly missed
3. **gt_object: category = chair** (and 36 more values): 53.0% vs 33.8%, 18% of object failures are excess
4. **gt_object: supercategory = furniture** (and 5 more values): 47.1% vs 33.5%, 16% of object failures are excess, mostly mislocalized
5. **prediction: category = book** (and 32 more values): 64.3% vs 41.0%, 13% of prediction failures are excess, mostly background
6. **prediction: supercategory = furniture** (and 6 more values): 54.6% vs 40.4%, 12% of prediction failures are excess, mostly mislocalized
7. **gt_object: n_objects (quintile) = 22 to 62** (and 2 more values): 45.7% vs 32.3%, 11% of object failures are excess (partly rel_area (quintile))
8. **prediction: clip_scene = indoor**: 47.6% vs 37.9%, 9% of prediction failures are excess

### 1. prediction: score (quintile) = 0.5 to 0.642
**79.7% wrong** vs 32.2% for the rest (2.47×, 95% CI 78.8%–80.5%). 8,459 predictions, 6,742 failures, **4,015 more than expected**.
- Other significantly worse values of `score (quintile)`: 0.642 to 0.8038 65.8% (n=8,460). All together: 6,561 excess failures = 37% of all prediction failures.
- Same group of predictions as: score_bin = 0.5-0.6
- Failure type: about the same mix as overall, so this slice fails *more often*, not *differently*. Mix: mislocalized 44%, background 41%, wrong_class 10%, duplicate 5%.
- Confounding: **holds** within every other attribute (25 tested; the strongest, category, still leaves +42.2% points of the +47.5% gap).
- What to check: Low-confidence region: this is a threshold choice, see *Calibration and threshold* below.
- Phoenix: `name == 'prediction' and metadata['score'] >= 0.5 and metadata['score'] <= 0.642 and annotations['correct'].label == 'incorrect'`

### 2. gt_object: size = small
**55.3% not found** vs 25.3% for the rest (2.18×, 95% CI 54.4%–56.2%). 11,889 objects, 6,570 failures, **3,558 more than expected**.
- Share of all object failures that this excess represents: 27%.
- Same group of objects as: rel_area (quintile) = 1e-05 to 0.0018, area_px (quintile) = 2.5 to 482.2
- Failure type: mostly **missed** (72% of failures here vs 53% overall, 1.4× over-represented). Mix: missed 72%, mislocalized 23%, confused 4%, claimed_by_neighbour 1%.
- Confounding: **holds** within every other attribute (23 tested; the strongest, n_objects_bucket, still leaves +27.9% points of the +29.9% gap).
- What to check: The model draws no box here at all. Check input resolution / small-object capacity, and whether lower-scored boxes exist (re-run with a lower score threshold to split 'never proposed' from 'scored too low').
- Phoenix: `name == 'gt_object' and metadata['size'] == 'small' and annotations['outcome'].label == 'missed'`

### 3. gt_object: category = chair
**53.0% not found** vs 33.8% for the rest (1.57×, 95% CI 50.8%–55.3%). 1,838 objects, 975 failures, **353 more than expected**.
- Other significantly worse values of `category`: knife 70.6% (n=337), handbag 69.8% (n=559), spoon 66.1% (n=257), bench 65.3% (n=432), scissors 63.2% (n=38), backpack 62.3% (n=387), book 60.2% (n=1,156), toothbrush 61.4% (n=57) (+28 more). All together: 2,381 excess failures = 18% of all object failures.
- Failure type: about the same mix as overall, so this slice fails *more often*, not *differently*. Mix: missed 52%, mislocalized 37%, confused 9%, claimed_by_neighbour 1%.
- Confounding: **holds** within every other attribute (23 tested; the strongest, n_classes (quintile), still leaves +14.9% points of the +19.2% gap).
- Phoenix: `name == 'gt_object' and metadata['category'] == 'chair' and annotations['correct'].label == 'incorrect'`

### 4. gt_object: supercategory = furniture
**47.1% not found** vs 33.5% for the rest (1.41×, 95% CI 45.5%–48.7%). 3,529 objects, 1,662 failures, **480 more than expected**.
- Other significantly worse values of `supercategory`: accessory 53.9% (n=1,958), indoor 49.7% (n=2,025), food 46.7% (n=2,892), outdoor 46.7% (n=1,334), kitchen 45.0% (n=3,764). All together: 2,162 excess failures = 16% of all object failures.
- Failure type: mostly **mislocalized** (41% of failures here vs 35% overall, 1.2× over-represented). Mix: missed 45%, mislocalized 41%, confused 13%, claimed_by_neighbour 1%.
- Confounding: **holds** within every other attribute (23 tested; the strongest, n_classes (quintile), still leaves +10.0% points of the +13.6% gap).
- What to check: A box is nearby but misplaced (IoU too low). Look at box quality: occlusion, truncation, unusual aspect ratios, or inconsistent box conventions in the labels.
- Phoenix: `name == 'gt_object' and metadata['supercategory'] == 'furniture' and annotations['outcome'].label == 'mislocalized'`

### 5. prediction: category = book
**64.3% wrong** vs 41.0% for the rest (1.57×, 95% CI 61.6%–66.9%). 1,288 predictions, 828 failures, **300 more than expected**.
- Other significantly worse values of `category`: carrot 65.2% (n=540), knife 65.4% (n=286), broccoli 65.1% (n=501), apple 64.6% (n=277), handbag 63.9% (n=468), backpack 63.7% (n=402), toothbrush 63.3% (n=60), scissors 63.2% (n=38) (+24 more). All together: 2,340 excess failures = 13% of all prediction failures.
- Failure type: mostly **background** (58% of failures here vs 39% overall, 1.5× over-represented). Mix: background 58%, mislocalized 37%, duplicate 4%, wrong_class 2%.
- Confounding: **holds** within every other attribute (22 tested; the strongest, clip_setting, still leaves +18.2% points of the +23.3% gap).
- What to check: Boxes where nothing is labelled. Check for unlabelled objects first (label pipeline), then treat the rest as false alarms (threshold, hard negatives).
- Phoenix: `name == 'prediction' and metadata['category'] == 'book' and annotations['outcome'].label == 'background'`

### 6. prediction: supercategory = furniture
**54.6% wrong** vs 40.4% for the rest (1.35×, 95% CI 53.0%–56.1%). 4,109 predictions, 2,242 failures, **584 more than expected**.
- Other significantly worse values of `supercategory`: indoor 57.5% (n=2,397), food 54.7% (n=3,403), accessory 53.7% (n=1,952), appliance 52.6% (n=763), outdoor 49.0% (n=1,395), kitchen 48.2% (n=3,993). All together: 2,182 excess failures = 12% of all prediction failures.
- Failure type: mostly **mislocalized** (56% of failures here vs 46% overall, 1.2× over-represented). Mix: mislocalized 56%, background 30%, wrong_class 10%, duplicate 4%.
- Confounding: **holds** within every other attribute (22 tested; the strongest, clip_setting, still leaves +9.7% points of the +14.2% gap).
- What to check: A box is nearby but misplaced (IoU too low). Look at box quality: occlusion, truncation, unusual aspect ratios, or inconsistent box conventions in the labels.
- Phoenix: `name == 'prediction' and metadata['supercategory'] == 'furniture' and annotations['outcome'].label == 'mislocalized'`

### 7. gt_object: n_objects (quintile) = 22 to 62
**45.7% not found** vs 32.3% for the rest (1.42×, 95% CI 44.6%–46.9%). 7,025 objects, 3,212 failures, **946 more than expected**.
- Other significantly worse values of `n_objects (quintile)`: 15 to 22 37.7% (n=7,772), 10 to 15 36.3% (n=7,566). All together: 1,385 excess failures = 11% of all object failures.
- Failure type: about the same mix as overall, so this slice fails *more often*, not *differently*. Mix: missed 61%, mislocalized 32%, confused 6%, claimed_by_neighbour 1%.
- Confounding: **partly explained by rel_area (quintile)**: within rel_area (quintile) the gap shrinks from +13.5% to +5.5% points (41% remains).
- Phoenix: `name == 'gt_object' and metadata['n_objects'] > 22 and metadata['n_objects'] <= 62 and annotations['correct'].label == 'incorrect'`

### 8. prediction: clip_scene = indoor
**47.6% wrong** vs 37.9% for the rest (1.25×, 95% CI 46.8%–48.3%). 16,726 predictions, 7,956 failures, **1,613 more than expected**.
- Share of all prediction failures that this excess represents: 9%.
- Failure type: about the same mix as overall, so this slice fails *more often*, not *differently*. Mix: mislocalized 45%, background 41%, wrong_class 10%, duplicate 4%.
- Confounding: **holds** within every other attribute (21 tested; the strongest, sharpness (quintile), still leaves +8.8% points of the +9.6% gap).
- Phoenix: `name == 'prediction' and metadata['clip_scene'] == 'indoor' and annotations['correct'].label == 'incorrect'`

Smaller problems (also significant, not detailed): prediction: clip_setting = food (50.2% vs 40.0%, 9%); gt_object: n_classes (quintile) = 6 to 14 (45.0% vs 32.9%, 9%); gt_object: clip_scene = indoor (39.2% vs 32.0%, 8%); prediction: size = small (51.2% vs 38.5%, 8%); prediction: position = middle (43.2% vs 40.1%, 7%); prediction: n_objects (quintile) = 0 to 4 (47.2% vs 40.2%, 6%); gt_object: position = edge (42.7% vs 33.6%, 5%); prediction: center_dist (quintile) = 0.615 to 0.965 (48.2% vs 40.1%, 5%); prediction: n_objects_bucket = 4-7 (45.3% vs 40.9%, 4%); gt_object: brightness (quintile) = 3.17 to 90.05 (37.9% vs 34.0%, 3%); prediction: n_preds (quintile) = 22 to 59 (45.8% vs 40.8%, 3%); gt_object: img_height (quintile) = 428 to 480 (36.3% vs 34.1%, 2%); prediction: img_width (quintile) = 200 to 480 (44.4% vs 41.1%, 2%); gt_object: contrast (quintile) = 64.7 to 71.76 (36.5% vs 34.3%, 1%); prediction: contrast (quintile) = 64.72 to 71.89 (42.8% vs 41.5%, 1%).

## 3. Looks like a problem, but is explained by something else

These slices fail more often overall, but the gap mostly disappears once you compare like with like.

- **gt_object: truncated = False** (35.8% vs 29.3%): within *rel_area (quintile)* the gap falls from +6.5% to -5.8% points.
- **gt_object: n_objects_bucket = 16+** (41.5% vs 30.4%): within *rel_area (quintile)* the gap falls from +11.1% to +2.1% points.
- **gt_object: overlap (quintile) = 0 to 0.041** (38.7% vs 32.1%): within *rel_area (quintile)* the gap falls from +6.6% to -3.9% points.
- **prediction: aspect_ratio (quintile) = 1.574 to 21.57** (51.0% vs 39.4%): within *category* the gap falls from +11.6% to +3.5% points.
- **gt_object: aspect_ratio (quintile) = 1.575 to 46.7** (43.6% vs 32.5%): within *category* the gap falls from +11.1% to +3.3% points.
- **prediction: sharpness (quintile) = 10.92 to 872** (45.3% vs 40.8%): within *clip_setting* the gap falls from +4.4% to +1.2% points.
- **gt_object: clip_setting = home** (40.3% vs 33.9%): within *category* the gap falls from +6.3% to -3.9% points.
- **gt_object: clip_focus = blurry** (36.1% vs 34.0%): within *n_classes (quintile)* the gap falls from +2.1% to +0.7% points.
- **gt_object: n_preds (quintile) = 21 to 59** (37.2% vs 34.1%): within *n_objects_bucket* the gap falls from +3.1% to -6.8% points.
- **prediction: img_megapixels (quintile) = 0.3 to 0.307** (42.9% vs 41.2%): within *clip_setting* the gap falls from +1.6% to +0.1% points.
- **prediction: n_classes (quintile) = 0 to 2** (42.9% vs 41.2%): within *n_objects_bucket* the gap falls from +1.7% to -3.5% points.
- **prediction: truncated = True** (44.1% vs 41.4%): within *position* the gap falls from +2.7% to -0.3% points.

## 4. Calibration and threshold

`pred_units`: share correct by `score`.

| score | n | correct | most common error |
|---|---|---|---|
| (0.499, 0.6] | 6105 | 19.2% | mislocalized (43%) |
| (0.6, 0.7] | 5341 | 27.0% | mislocalized (45%) |
| (0.7, 0.8] | 5256 | 36.5% | mislocalized (47%) |
| (0.8, 0.9] | 5959 | 52.2% | mislocalized (46%) |
| (0.9, 1.0] | 19616 | 86.6% | mislocalized (51%) |

Threshold trade-off (recall = share of ground truth still found):

| threshold | kept | precision | recall | f1 |
|---|---|---|---|---|
| 0.5 | 42277 | 58.3% | 65.2% | 61.6% |
| 0.55 | 39122 | 61.5% | 63.8% | 62.6% |
| 0.6 | 36179 | 64.8% | 62.1% | 63.5% |
| 0.65 | 33366 | 68.3% | 60.4% | 64.1% |
| 0.7 | 30836 | 71.4% | 58.3% | 64.2% |
| 0.75 | 28240 | 75.0% | 56.1% | 64.2% |
| 0.8 | 25584 | 78.6% | 53.2% | 63.5% |
| 0.85 | 22798 | 82.5% | 49.8% | 62.1% |
| 0.9 | 19627 | 86.6% | 45.0% | 59.2% |
| 0.95 | 15425 | 91.4% | 37.3% | 53.0% |

Best F1 at score ≥ **0.70** (precision 71.4%, recall 58.3%). Whether that is the right point depends on whether misses or false alarms cost more.

> Note: the lowest score is 0.50, so predictions were already cut at that threshold. Objects "missed" may have had a lower-scored box the file does not contain.

## 5. Class confusions

1,379 gt_units rows are in the right place with the wrong class. Most common pairs (`category` → `pred_label`):

| category | pred_label | count |
|---|---|---|
| truck | car | 85 |
| car | truck | 35 |
| chair | couch | 29 |
| handbag | backpack | 28 |
| chair | person | 25 |
| cup | bowl | 24 |
| bottle | cup | 23 |
| couch | chair | 23 |
| wine glass | cup | 18 |
| cup | wine glass | 17 |
| bus | truck | 15 |
| cup | bottle | 15 |
| backpack | handbag | 15 |
| bowl | cup | 15 |
| hot dog | sandwich | 14 |

Both directions combined: car ↔ truck (120), chair ↔ couch (52), backpack ↔ handbag (43), bowl ↔ cup (39), bottle ↔ cup (38), cup ↔ wine glass (35), chair ↔ person (30), bus ↔ truck (23).

Pairs that go both ways are often ambiguous in the labels too; check them with the label pipeline.

## 6. Possible label issues

937 of 6,951 detections on unlabelled regions score ≥ 0.9. Confident detections with nothing labelled underneath are often objects missing from the labels, not model errors.

| category | count |
|---|---|
| person | 196 |
| bottle | 72 |
| car | 48 |
| chair | 47 |
| cup | 46 |
| traffic light | 33 |
| sheep | 27 |
| broccoli | 26 |
| book | 21 |
| bowl | 20 |

Phoenix: `name == 'prediction' and annotations['outcome'].label == 'background' and metadata['score'] >= 0.9`

## 7. Localization sensitivity

4,533 gt_units rows have a box nearby that overlaps too little. Their best IoU:

| IoU | count |
|---|---|
| (0.0, 0.1] | 9 |
| (0.1, 0.2] | 1435 |
| (0.2, 0.3] | 1028 |
| (0.3, 0.4] | 964 |
| (0.4, 0.5] | 1097 |

Loosening the match would raise the correct rate from 65.2% to 67.5% at IoU 0.4, or 69.3% at IoU 0.3. Small gain: when boxes are off, they are well off.

## 8. Where failures concentrate

- `gt_units`: the worst 10% of images hold **46%** of failures; 115 of 5,150 images have none of their objects correct. Worst images: 000000435081 (52), 000000303566 (31), 000000036494 (31), 000000416991 (30), 000000316666 (29).
- `pred_units`: the worst 10% of images hold **36%** of failures; 118 of 5,153 images have none of their predictions correct. Worst images: 000000031248 (46), 000000546556_wrong (42), 000000546556 (42), 000000498919 (37), 000000076468 (36).

## 9. Every attribute: its worst value

The single worst value of each attribute, significant or not (p_adj = Benjamini-Hochberg corrected).

**`gt_object`** (overall 34.8% not found)

| attribute | worst | n | fail | rest | lift | excess | significant |
|---|---|---|---|---|---|---|---|
| size | small | 11,889 | 55.3% | 25.3% | 2.18 | 3,558 | yes |
| rel_area (quintile) | 1e-05 to 0.0018 | 7,565 | 61.0% | 28.2% | 2.16 | 2,480 | yes |
| area_px (quintile) | 2.5 to 482.2 | 7,552 | 60.9% | 28.2% | 2.16 | 2,469 | yes |
| truncated | False | 31,608 | 35.8% | 29.3% | 1.22 | 2,060 | yes |
| n_objects_bucket | 16+ | 14,797 | 41.5% | 30.4% | 1.37 | 1,644 | yes |
| clip_scene | indoor | 14,435 | 39.2% | 32.0% | 1.23 | 1,048 | yes |
| overlap (quintile) | 0 to 0.041 | 15,194 | 38.7% | 32.1% | 1.21 | 1,007 | yes |
| n_objects (quintile) | 22 to 62 | 7,025 | 45.7% | 32.3% | 1.42 | 946 | yes |
| aspect_ratio (quintile) | 1.575 to 46.7 | 7,548 | 43.6% | 32.5% | 1.34 | 838 | yes |
| n_classes (quintile) | 6 to 14 | 5,749 | 45.0% | 32.9% | 1.37 | 693 | yes |
| aspect | wide | 4,815 | 47.0% | 33.0% | 1.43 | 677 | yes |
| center_dist (quintile) | 0.614 to 0.977 | 7,543 | 41.6% | 33.1% | 1.26 | 644 | yes |
| img_width (quintile) | 500 to 640 | 26,661 | 35.3% | 33.4% | 1.06 | 500 | yes |
| position | edge | 4,997 | 42.7% | 33.6% | 1.27 | 457 | yes |
| supercategory | accessory | 1,958 | 53.9% | 33.7% | 1.60 | 395 | yes |
| clip_setting | home | 4,850 | 40.3% | 33.9% | 1.19 | 308 | yes |
| clip_focus | blurry | 14,453 | 36.1% | 34.0% | 1.06 | 305 | yes |
| brightness (quintile) | 3.17 to 90.05 | 7,556 | 37.9% | 34.0% | 1.11 | 292 | yes |
| img_height (quintile) | 428 to 480 | 11,455 | 36.3% | 34.1% | 1.07 | 257 | yes |
| img_megapixels (quintile) | 0.303 to 0.307 | 11,974 | 36.2% | 34.1% | 1.06 | 256 | yes |
| n_preds (quintile) | 21 to 59 | 7,497 | 37.2% | 34.1% | 1.09 | 232 | yes |
| contrast (quintile) | 64.7 to 71.76 | 7,556 | 36.5% | 34.3% | 1.06 | 166 | yes |
| category | knife | 337 | 70.6% | 34.4% | 2.05 | 122 | yes |
| img_orientation | square | 1,699 | 39.7% | 34.5% | 1.15 | 87 | yes |
| clip_lighting | dark | 2,278 | 36.7% | 34.6% | 1.06 | 46 | no |
| sharpness (quintile) | 2578 to 4270 | 7,548 | 35.0% | 34.7% | 1.01 | 27 | no |
| clip_color | black and white | 1,225 | 35.1% | 34.7% | 1.01 | 4 | no |

**`prediction`** (overall 41.7% wrong)

| attribute | worst | n | fail | rest | lift | excess | significant |
|---|---|---|---|---|---|---|---|
| score (quintile) | 0.5 to 0.642 | 8,459 | 79.7% | 32.2% | 2.47 | 4,015 | yes |
| score_bin | 0.5-0.6 | 6,102 | 80.8% | 35.1% | 2.30 | 2,786 | yes |
| clip_scene | indoor | 16,726 | 47.6% | 37.9% | 1.25 | 1,613 | yes |
| size | small | 10,865 | 51.2% | 38.5% | 1.33 | 1,386 | yes |
| area_px (quintile) | 9.6 to 687.9 | 8,456 | 53.1% | 38.9% | 1.37 | 1,203 | yes |
| rel_area (quintile) | 4e-05 to 0.0026 | 8,469 | 52.7% | 39.0% | 1.35 | 1,164 | yes |
| aspect_ratio (quintile) | 1.574 to 21.57 | 8,452 | 51.0% | 39.4% | 1.29 | 980 | yes |
| clip_setting | food | 7,174 | 50.2% | 40.0% | 1.26 | 733 | yes |
| center_dist (quintile) | 0.615 to 0.965 | 8,376 | 48.2% | 40.1% | 1.20 | 677 | yes |
| aspect | wide | 5,174 | 53.1% | 40.2% | 1.32 | 667 | yes |
| clip_color | color | 40,948 | 41.8% | 40.2% | 1.04 | 659 | no |
| n_objects (quintile) | 0 to 4 | 9,059 | 47.2% | 40.2% | 1.17 | 632 | yes |
| position | edge | 5,598 | 49.5% | 40.6% | 1.22 | 501 | yes |
| supercategory | indoor | 2,397 | 57.5% | 40.8% | 1.41 | 400 | yes |
| n_preds (quintile) | 22 to 59 | 7,743 | 45.8% | 40.8% | 1.12 | 389 | yes |
| sharpness (quintile) | 10.92 to 872 | 8,487 | 45.3% | 40.8% | 1.11 | 376 | yes |
| img_width (quintile) | 200 to 480 | 8,555 | 44.4% | 41.1% | 1.08 | 289 | yes |
| img_orientation | portrait | 8,802 | 44.0% | 41.2% | 1.07 | 248 | yes |
| img_megapixels (quintile) | 0.3 to 0.307 | 13,526 | 42.9% | 41.2% | 1.04 | 223 | yes |
| img_height (quintile) | 612 to 640 | 7,963 | 43.3% | 41.4% | 1.05 | 152 | yes |
| truncated | True | 5,169 | 44.1% | 41.4% | 1.07 | 142 | yes |
| brightness (quintile) | 90.84 to 106.4 | 8,468 | 43.0% | 41.4% | 1.04 | 136 | yes |
| category | carrot | 540 | 65.2% | 41.4% | 1.57 | 128 | yes |
| n_classes (quintile) | 6 to 14 | 5,615 | 43.7% | 41.4% | 1.05 | 125 | yes |
| contrast (quintile) | 64.72 to 71.89 | 8,478 | 42.8% | 41.5% | 1.03 | 113 | yes |
| clip_lighting | dark | 2,622 | 45.0% | 41.5% | 1.08 | 90 | yes |
| clip_focus | blurry | 15,946 | 42.0% | 41.6% | 1.01 | 79 | no |
| n_objects_bucket | 0 | 51 | 100.0% | 41.7% | 2.40 | 30 | yes |

## 10. Caveats

- Everything here is correlation over this evaluation set. A slice tells you where to look, not why.
- Confounding is checked one attribute at a time; two attributes together can still explain a slice.
- Slices smaller than the minimum size are not reported; small slices have wide confidence intervals.
- `clip_*` attributes are zero-shot CLIP guesses per image and are noisy (especially focus/lighting).
