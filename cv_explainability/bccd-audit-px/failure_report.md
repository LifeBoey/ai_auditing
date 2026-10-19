# Model failure report

Generated 2026-10-16 15:50 from `bccd-audit-px` (gt_units.csv, pred_units.csv).

## 1. Overview (what Phoenix shows)

Each table is one kind of unit, and each row of it one span in Phoenix. The `correct` annotation summary in Phoenix is the accuracy column below; filter spans and it updates for that slice.

| unit | rows | correct | failures |
|---|---|---|---|
| `gt_object` (gt_units.csv) | 4,886 | 80.7% | missed 70%, mislocalized 27%, claimed_by_neighbour 3%, confused 0% |
| `prediction` (pred_units.csv) | 4,391 | 89.8% | background 73%, mislocalized 26%, wrong_class 1%, duplicate 0% |

- **gt_object**: 80.7% of 4,886 objects are correct; the commonest failure is *missed* (70% of failures), then *mislocalized* (27%). Biggest problem area: `category = RBC` (21.9% not found vs 4.8%).
- **prediction**: 89.8% of 4,391 predictions are correct; the commonest failure is *background* (73% of failures), then *mislocalized* (26%). Biggest problem area: `score (quintile) = 0.5001 to 0.7201` (29.6% wrong vs 5.3%).

## 2. Main problems

One entry per attribute, ranked by its excess failures (failures beyond what the overall rate predicts, summed over all its significantly worse values) as a share of that unit's failures. Slices that another attribute fully explains are in section 3 instead.

1. **gt_object: category = RBC**: 21.9% vs 4.8%, 75% of object failures are excess
2. **prediction: score (quintile) = 0.5001 to 0.7201** (and 1 more values): 29.6% vs 5.3%, 58% of prediction failures are excess, mostly mislocalized
3. **prediction: score_bin = 0.5-0.6** (and 2 more values): 36.4% vs 7.7%, 50% of prediction failures are excess, mostly mislocalized
4. **prediction: n_objects (quintile) = 1 to 11**: 23.1% vs 6.3%, 38% of prediction failures are excess (partly n_objects_bucket)
5. **prediction: category = RBC**: 10.9% vs 6.9%, 32% of prediction failures are excess (partly n_classes)
6. **gt_object: size = medium**: 32.1% vs 13.5%, 30% of object failures are excess
7. **gt_object: n_objects_bucket = 16+**: 26.9% vs 14.4%, 26% of object failures are excess, mostly mislocalized
8. **prediction: n_objects_bucket = 1** (and 2 more values): 87.5% vs 9.3%, 22% of prediction failures are excess, mostly background

### 1. gt_object: category = RBC
**21.9% not found** vs 4.8% for the rest (4.58×, 95% CI 20.6%–23.1%). 4,153 objects, 908 failures, **710 more than expected**.
- Share of all object failures that this excess represents: 75%.
- Failure type: about the same mix as overall, so this slice fails *more often*, not *differently*. Mix: missed 69%, mislocalized 28%, claimed_by_neighbour 3%.
- Confounding: **holds** within every other attribute (10 tested; the strongest, center_dist (quintile), still leaves +15.9% points of the +17.1% gap).
- Phoenix: `name == 'gt_object' and metadata['category'] == 'RBC' and annotations['correct'].label == 'incorrect'`

### 2. prediction: score (quintile) = 0.5001 to 0.7201
**29.6% wrong** vs 5.3% for the rest (5.56×, 95% CI 26.7%–32.7%). 881 predictions, 261 failures, **214 more than expected**.
- Other significantly worse values of `score (quintile)`: 0.7201 to 0.8764 14.4% (n=877). All together: 260 excess failures = 58% of all prediction failures.
- Failure type: mostly **mislocalized** (32% of failures here vs 26% overall, 1.3× over-represented). Mix: background 66%, mislocalized 32%, wrong_class 1%, duplicate 1%.
- Confounding: **holds** within every other attribute (12 tested; the strongest, n_classes, still leaves +23.3% points of the +24.3% gap).
- What to check: Low-confidence region: this is a threshold choice, see *Calibration and threshold* below.
- Phoenix: `name == 'prediction' and metadata['score'] >= 0.5001 and metadata['score'] <= 0.7201 and annotations['outcome'].label == 'mislocalized'`

### 3. prediction: score_bin = 0.5-0.6
**36.4% wrong** vs 7.7% for the rest (4.72×, 95% CI 31.7%–41.3%). 382 predictions, 139 failures, **110 more than expected**.
- Other significantly worse values of `score_bin`: 0.6-0.7 26.3% (n=422), 0.7-0.8 17.6% (n=455). All together: 222 excess failures = 50% of all prediction failures.
- Failure type: mostly **mislocalized** (34% of failures here vs 26% overall, 1.3× over-represented). Mix: background 65%, mislocalized 34%, wrong_class 1%.
- Confounding: **holds** within every other attribute (12 tested; the strongest, n_objects_bucket, still leaves +27.0% points of the +28.7% gap).
- What to check: Low-confidence region: this is a threshold choice, see *Calibration and threshold* below.
- Phoenix: `name == 'prediction' and metadata['score_bin'] == '0.5-0.6' and annotations['outcome'].label == 'mislocalized'`

### 4. prediction: n_objects (quintile) = 1 to 11
**23.1% wrong** vs 6.3% for the rest (3.66×, 95% CI 20.6%–25.7%). 1,023 predictions, 236 failures, **172 more than expected**.
- Share of all prediction failures that this excess represents: 38%.
- Failure type: about the same mix as overall, so this slice fails *more often*, not *differently*. Mix: background 83%, mislocalized 16%, wrong_class 0%.
- Confounding: **partly explained by n_objects_bucket**: within n_objects_bucket the gap shrinks from +16.8% to +6.7% points (40% remains).
- Phoenix: `name == 'prediction' and metadata['n_objects'] >= 1 and metadata['n_objects'] <= 11 and annotations['correct'].label == 'incorrect'`

### 5. prediction: category = RBC
**10.9% wrong** vs 6.9% for the rest (1.57×, 95% CI 9.9%–11.9%). 3,641 predictions, 396 failures, **144 more than expected**.
- Share of all prediction failures that this excess represents: 32%.
- Failure type: about the same mix as overall, so this slice fails *more often*, not *differently*. Mix: background 70%, mislocalized 29%, duplicate 1%, wrong_class 0%.
- Confounding: **partly explained by n_classes**: within n_classes the gap shrinks from +3.9% to +1.5% points (38% remains).
- Phoenix: `name == 'prediction' and metadata['category'] == 'RBC' and annotations['correct'].label == 'incorrect'`

### 6. gt_object: size = medium
**32.1% not found** vs 13.5% for the rest (2.38×, 95% CI 29.8%–34.5%). 1,524 objects, 489 failures, **283 more than expected**.
- Share of all object failures that this excess represents: 30%.
- Same group of objects as: area_px (quintile) = 437 to 8096, rel_area (quintile) = 0.00142 to 0.02635
- Failure type: about the same mix as overall, so this slice fails *more often*, not *differently*. Mix: missed 73%, mislocalized 24%, claimed_by_neighbour 2%.
- Confounding: **holds** within every other attribute (10 tested; the strongest, category, still leaves +15.5% points of the +18.6% gap).
- Phoenix: `name == 'gt_object' and metadata['size'] == 'medium' and annotations['correct'].label == 'incorrect'`

### 7. gt_object: n_objects_bucket = 16+
**26.9% not found** vs 14.4% for the rest (1.87×, 95% CI 24.9%–28.9%). 1,932 objects, 519 failures, **242 more than expected**.
- Share of all object failures that this excess represents: 26%.
- Failure type: mostly **mislocalized** (34% of failures here vs 27% overall, 1.2× over-represented). Mix: missed 63%, mislocalized 34%, claimed_by_neighbour 4%.
- Confounding: **holds** within every other attribute (11 tested; the strongest, rel_area (quintile), still leaves +10.9% points of the +12.5% gap).
- What to check: A box is nearby but misplaced (IoU too low). Look at box quality: occlusion, truncation, unusual aspect ratios, or inconsistent box conventions in the labels.
- Phoenix: `name == 'gt_object' and metadata['n_objects_bucket'] == '16+' and annotations['outcome'].label == 'mislocalized'`

### 8. prediction: n_objects_bucket = 1
**87.5% wrong** vs 9.3% for the rest (9.36×, 95% CI 75.3%–94.1%). 48 predictions, 42 failures, **38 more than expected**.
- Other significantly worse values of `n_objects_bucket`: 2-3 64.6% (n=48), 4-7 37.0% (n=127). All together: 99 excess failures = 22% of all prediction failures.
- Failure type: mostly **background** (100% of failures here vs 73% overall, 1.4× over-represented). Mix: background 100%.
- Confounding: **holds** within every other attribute (10 tested; the strongest, area_px (quintile), still leaves +67.4% points of the +78.2% gap).
- What to check: Boxes where nothing is labelled. Check for unlabelled objects first (label pipeline), then treat the rest as false alarms (threshold, hard negatives).
- Phoenix: `name == 'prediction' and metadata['n_objects_bucket'] == '1' and annotations['outcome'].label == 'background'`

Smaller problems (also significant, not detailed): gt_object: overlap (quintile) = 0.109 to 0.933 (34.4% vs 15.5%, 20%); gt_object: n_objects (quintile) = 18 to 30 (32.0% vs 16.7%, 19%); gt_object: truncated = True (30.8% vs 15.6%, 19%); gt_object: center_dist (quintile) = 0.65 to 0.94 (29.6% vs 16.8%, 18%); gt_object: n_preds (quintile) = 3 to 10 (28.0% vs 16.7%, 13%); gt_object: aspect_ratio (quintile) = 0.333 to 0.924 (26.0% vs 17.6%, 13%); prediction: center_dist (quintile) = 0.64 to 0.939 (13.6% vs 9.3%, 8%).

## 3. Looks like a problem, but is explained by something else

These slices fail more often overall, but the gap mostly disappears once you compare like with like.

- **gt_object: n_classes = 3** (21.4% vs 16.3%): within *n_objects_bucket* the gap falls from +5.1% to +0.0% points.
- **prediction: size = medium** (14.3% vs 8.8%): within *category* the gap falls from +5.6% to +0.4% points.
- **prediction: n_classes = 2** (11.9% vs 9.0%): within *n_objects (quintile)* the gap falls from +2.9% to -2.3% points.
- **prediction: n_preds (quintile) = 3 to 10** (13.0% vs 9.5%): within *n_objects_bucket* the gap falls from +3.5% to -6.2% points.

## 4. Calibration and threshold

`pred_units`: share correct by `score`.

| score | n | correct | most common error |
|---|---|---|---|
| (0.499, 0.6] | 382 | 63.6% | background (65%) |
| (0.6, 0.7] | 422 | 73.7% | background (68%) |
| (0.7, 0.8] | 456 | 82.2% | background (75%) |
| (0.8, 0.9] | 732 | 90.0% | background (86%) |
| (0.9, 1.0] | 2399 | 98.2% | background (82%) |

Threshold trade-off (recall = share of ground truth still found):

| threshold | kept | precision | recall | f1 |
|---|---|---|---|---|
| 0.5 | 4391 | 89.8% | 80.7% | 85.0% |
| 0.55 | 4200 | 91.1% | 78.3% | 84.3% |
| 0.6 | 4010 | 92.3% | 75.7% | 83.2% |
| 0.65 | 3807 | 93.5% | 72.9% | 81.9% |
| 0.7 | 3587 | 94.5% | 69.4% | 80.0% |
| 0.75 | 3382 | 95.2% | 65.9% | 77.9% |
| 0.8 | 3132 | 96.2% | 61.7% | 75.2% |
| 0.85 | 2822 | 97.1% | 56.1% | 71.1% |
| 0.9 | 2399 | 98.2% | 48.2% | 64.7% |
| 0.95 | 1723 | 99.2% | 35.0% | 51.7% |

Best F1 at score ≥ **0.50** (precision 89.8%, recall 80.7%). Whether that is the right point depends on whether misses or false alarms cost more.

> Note: the lowest score is 0.50, so predictions were already cut at that threshold. Objects "missed" may have had a lower-scored box the file does not contain.

## 5. Class confusions

3 gt_units rows are in the right place with the wrong class. Most common pairs (`category` → `pred_label`):

| category | pred_label | count |
|---|---|---|
| Platelets | WBC | 2 |
| WBC | Platelets | 1 |

Both directions combined: Platelets ↔ WBC (3).

Pairs that go both ways are often ambiguous in the labels too; check them with the label pipeline.

## 6. Possible label issues

36 of 326 detections on unlabelled regions score ≥ 0.9. Confident detections with nothing labelled underneath are often objects missing from the labels, not model errors.

| category | count |
|---|---|
| RBC | 30 |
| Platelets | 5 |
| WBC | 1 |

Phoenix: `name == 'prediction' and annotations['outcome'].label == 'background' and metadata['score'] >= 0.9`

## 7. Localization sensitivity

255 gt_units rows have a box nearby that overlaps too little. Their best IoU:

| IoU | count |
|---|---|
| (0.0, 0.1] | 0 |
| (0.1, 0.2] | 136 |
| (0.2, 0.3] | 72 |
| (0.3, 0.4] | 28 |
| (0.4, 0.5] | 19 |

Loosening the match would raise the correct rate from 80.7% to 81.1% at IoU 0.4, or 81.7% at IoU 0.3. Small gain: when boxes are off, they are well off.

## 8. Where failures concentrate

- `gt_units`: the worst 10% of images hold **33%** of failures; 0 of 364 images have none of their objects correct. Worst images: BloodImage_00094 (13), BloodImage_00045 (12), BloodImage_00030 (12), BloodImage_00038 (12), BloodImage_00031 (11).
- `pred_units`: the worst 10% of images hold **40%** of failures; 0 of 364 images have none of their predictions correct. Worst images: BloodImage_00330 (13), BloodImage_00250 (10), BloodImage_00322 (9), BloodImage_00268 (8), BloodImage_00327 (8).

## 9. Every attribute: its worst value

The single worst value of each attribute, significant or not (p_adj = Benjamini-Hochberg corrected).

**`gt_object`** (overall 19.3% not found)

| attribute | worst | n | fail | rest | lift | excess | significant |
|---|---|---|---|---|---|---|---|
| category | RBC | 4,153 | 21.9% | 4.8% | 4.58 | 710 | yes |
| size | medium | 1,524 | 32.1% | 13.5% | 2.38 | 283 | yes |
| n_objects_bucket | 16+ | 1,932 | 26.9% | 14.4% | 1.87 | 242 | yes |
| overlap (quintile) | 0.109 to 0.933 | 977 | 34.4% | 15.5% | 2.21 | 184 | yes |
| truncated | True | 1,182 | 30.8% | 15.6% | 1.97 | 179 | yes |
| area_px (quintile) | 437 to 8096 | 981 | 33.8% | 15.6% | 2.16 | 179 | yes |
| rel_area (quintile) | 0.00142 to 0.02635 | 981 | 33.8% | 15.6% | 2.16 | 179 | yes |
| n_classes | 3 | 2,854 | 21.4% | 16.3% | 1.31 | 145 | yes |
| n_preds (quintile) | 3 to 10 | 1,116 | 28.0% | 16.7% | 1.68 | 127 | yes |
| n_objects (quintile) | 18 to 30 | 820 | 32.0% | 16.7% | 1.91 | 125 | yes |
| center_dist (quintile) | 0.65 to 0.94 | 971 | 29.6% | 16.8% | 1.76 | 124 | yes |
| position | edge | 881 | 29.1% | 17.2% | 1.69 | 105 | yes |
| aspect_ratio (quintile) | 0.333 to 0.924 | 982 | 26.0% | 17.6% | 1.47 | 82 | yes |
| aspect | squarish | 4,859 | 19.1% | 51.9% | 0.37 | -1,590 | no |

**`prediction`** (overall 10.2% wrong)

| attribute | worst | n | fail | rest | lift | excess | significant |
|---|---|---|---|---|---|---|---|
| aspect | squarish | 4,388 | 10.2% | 0.0% | inf | 448 | no |
| score (quintile) | 0.5001 to 0.7201 | 881 | 29.6% | 5.3% | 5.56 | 214 | yes |
| n_objects (quintile) | 1 to 11 | 1,023 | 23.1% | 6.3% | 3.66 | 172 | yes |
| category | RBC | 3,641 | 10.9% | 6.9% | 1.57 | 144 | yes |
| score_bin | 0.5-0.6 | 382 | 36.4% | 7.7% | 4.72 | 110 | yes |
| size | medium | 1,137 | 14.3% | 8.8% | 1.64 | 63 | yes |
| area_px (quintile) | 886.3 to 8712 | 879 | 15.5% | 8.9% | 1.74 | 58 | yes |
| rel_area (quintile) | 0.00288 to 0.02836 | 879 | 15.5% | 8.9% | 1.74 | 58 | yes |
| n_classes | 1 | 90 | 61.1% | 9.1% | 6.69 | 47 | yes |
| n_objects_bucket | 1 | 48 | 87.5% | 9.3% | 9.36 | 38 | yes |
| center_dist (quintile) | 0.64 to 0.939 | 872 | 13.6% | 9.3% | 1.46 | 37 | yes |
| n_preds (quintile) | 3 to 10 | 923 | 13.0% | 9.5% | 1.37 | 33 | yes |
| position | edge | 720 | 13.2% | 9.6% | 1.37 | 26 | yes |
| aspect_ratio (quintile) | 0.508 to 0.953 | 884 | 11.9% | 9.8% | 1.21 | 19 | no |
| truncated | True | 558 | 12.4% | 9.9% | 1.25 | 14 | no |

## 10. Caveats

- Everything here is correlation over this evaluation set. A slice tells you where to look, not why.
- Confounding is checked one attribute at a time; two attributes together can still explain a slice.
- Slices smaller than the minimum size are not reported; small slices have wide confidence intervals.
