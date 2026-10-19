# Model failure report

Generated 2026-10-16 09:48 from `bcdd-audit-px` (gt_units.csv, pred_units.csv).

## 1. Overview (what Phoenix shows)

Each table is one kind of unit, and each row of it one span in Phoenix. The `correct` annotation summary in Phoenix is the accuracy column below; filter spans and it updates for that slice.

| unit | rows | correct | failures |
|---|---|---|---|
| `gt_object` (gt_units.csv) | 4,886 | 99.3% | claimed_by_neighbour 68%, mislocalized 32% |
| `prediction` (pred_units.csv) | 19,644 | 24.7% | background 39%, wrong_class 36%, mislocalized 23%, duplicate 2% |

- **gt_object**: 99.3% of 4,886 objects are correct; the commonest failure is *claimed_by_neighbour* (68% of failures), then *mislocalized* (32%). Biggest problem area: `overlap (quintile) = 0.109 to 0.933` (3.3% not found vs 0.1%).
- **prediction**: 24.7% of 19,644 predictions are correct; the commonest failure is *background* (39% of failures), then *wrong_class* (36%). Biggest problem area: `score_bin = 0.5-0.6` (92.6% wrong vs 7.7%).

## 2. Main problems

One entry per attribute, ranked by its excess failures (failures beyond what the overall rate predicts, summed over all its significantly worse values) as a share of that unit's failures. Slices that another attribute fully explains are in section 3 instead.

1. **gt_object: overlap (quintile) = 0.109 to 0.933**: 3.3% vs 0.1%, 93% of object failures are excess
2. **prediction: score_bin = 0.5-0.6**: 92.6% vs 7.7%, 90% of prediction failures are excess
3. **gt_object: size = medium**: 1.5% vs 0.3%, 53% of object failures are excess, mostly mislocalized
4. **gt_object: n_objects_bucket = 16+**: 1.2% vs 0.4%, 46% of object failures are excess (partly overlap (quintile))
5. **gt_object: n_objects (quintile) = 18 to 30**: 2.0% vs 0.4%, 36% of object failures are excess, mostly mislocalized (partly overlap (quintile))
6. **gt_object: n_preds (quintile) = 66 to 100**: 1.6% vs 0.5%, 31% of object failures are excess, mostly mislocalized (partly overlap (quintile))
7. **prediction: score (quintile) = 0.001 to 0.0018** (and 2 more values): 100.0% vs 68.9%, 24% of prediction failures are excess, mostly wrong_class
8. **prediction: category = Platelets** (and 1 more values): 91.1% vs 71.2%, 9% of prediction failures are excess, mostly wrong_class

### 1. gt_object: overlap (quintile) = 0.109 to 0.933
**3.3% not found** vs 0.1% for the rest (64.02×, 95% CI 2.3%–4.6%). 977 objects, 32 failures, **32 more than expected**.
- Share of all object failures that this excess represents: 93%.
- Failure type: about the same mix as overall, so this slice fails *more often*, not *differently*. Mix: claimed_by_neighbour 72%, mislocalized 28%.
- Confounding: **holds** within every other attribute (12 tested; the strongest, n_objects (quintile), still leaves +3.2% points of the +3.2% gap).
- Phoenix: `name == 'gt_object' and metadata['overlap'] > 0.109 and metadata['overlap'] <= 0.933 and annotations['correct'].label == 'incorrect'`

### 2. prediction: score_bin = 0.5-0.6
**92.6% wrong** vs 7.7% for the rest (12.02×, 95% CI 92.2%–93.0%). 15,635 predictions, 14,483 failures, **13,278 more than expected**.
- Share of all prediction failures that this excess represents: 90%.
- Failure type: about the same mix as overall, so this slice fails *more often*, not *differently*. Mix: background 38%, wrong_class 37%, mislocalized 23%, duplicate 2%.
- Confounding: **holds** within every other attribute (12 tested; the strongest, n_objects_bucket, still leaves +83.3% points of the +84.9% gap).
- What to check: Low-confidence region: this is a threshold choice, see *Calibration and threshold* below.
- Phoenix: `name == 'prediction' and metadata['score_bin'] == '0.5-0.6' and annotations['correct'].label == 'incorrect'`

### 3. gt_object: size = medium
**1.5% not found** vs 0.3% for the rest (4.61×, 95% CI 1.0%–2.3%). 1,524 objects, 23 failures, **18 more than expected**.
- Share of all object failures that this excess represents: 53%.
- Same group of objects as: area_px (quintile) = 437 to 8096, rel_area (quintile) = 0.00142 to 0.02635
- Failure type: mostly **mislocalized** (43% of failures here vs 32% overall, 1.3× over-represented). Mix: claimed_by_neighbour 57%, mislocalized 43%.
- Confounding: **holds** within every other attribute (10 tested; the strongest, overlap (quintile), still leaves +1.1% points of the +1.2% gap).
- What to check: A box is nearby but misplaced (IoU too low). Look at box quality: occlusion, truncation, unusual aspect ratios, or inconsistent box conventions in the labels.
- Phoenix: `name == 'gt_object' and metadata['size'] == 'medium' and annotations['outcome'].label == 'mislocalized'`

### 4. gt_object: n_objects_bucket = 16+
**1.2% not found** vs 0.4% for the rest (3.20×, 95% CI 0.8%–1.8%). 1,932 objects, 23 failures, **16 more than expected**.
- Share of all object failures that this excess represents: 46%.
- Failure type: about the same mix as overall, so this slice fails *more often*, not *differently*. Mix: claimed_by_neighbour 65%, mislocalized 35%.
- Confounding: **partly explained by overlap (quintile)**: within overlap (quintile) the gap shrinks from +0.8% to +0.3% points (40% remains).
- Phoenix: `name == 'gt_object' and metadata['n_objects_bucket'] == '16+' and annotations['correct'].label == 'incorrect'`

### 5. gt_object: n_objects (quintile) = 18 to 30
**2.0% not found** vs 0.4% for the rest (4.41×, 95% CI 1.2%–3.1%). 820 objects, 16 failures, **12 more than expected**.
- Share of all object failures that this excess represents: 36%.
- Failure type: mostly **mislocalized** (44% of failures here vs 32% overall, 1.4× over-represented). Mix: claimed_by_neighbour 56%, mislocalized 44%.
- Confounding: **partly explained by overlap (quintile)**: within overlap (quintile) the gap shrinks from +1.5% to +0.8% points (50% remains).
- What to check: A box is nearby but misplaced (IoU too low). Look at box quality: occlusion, truncation, unusual aspect ratios, or inconsistent box conventions in the labels.
- Phoenix: `name == 'gt_object' and metadata['n_objects'] > 18 and metadata['n_objects'] <= 30 and annotations['outcome'].label == 'mislocalized'`

### 6. gt_object: n_preds (quintile) = 66 to 100
**1.6% not found** vs 0.5% for the rest (3.33×, 95% CI 1.0%–2.6%). 937 objects, 15 failures, **10 more than expected**.
- Share of all object failures that this excess represents: 31%.
- Failure type: mostly **mislocalized** (40% of failures here vs 32% overall, 1.2× over-represented). Mix: claimed_by_neighbour 60%, mislocalized 40%.
- Confounding: **partly explained by overlap (quintile)**: within overlap (quintile) the gap shrinks from +1.1% to +0.7% points (65% remains).
- What to check: A box is nearby but misplaced (IoU too low). Look at box quality: occlusion, truncation, unusual aspect ratios, or inconsistent box conventions in the labels.
- Phoenix: `name == 'gt_object' and metadata['n_preds'] > 66 and metadata['n_preds'] <= 100 and annotations['outcome'].label == 'mislocalized'`

### 7. prediction: score (quintile) = 0.001 to 0.0018
**100.0% wrong** vs 68.9% for the rest (1.45×, 95% CI 99.9%–100.0%). 4,028 predictions, 4,028 failures, **1,252 more than expected**.
- Other significantly worse values of `score (quintile)`: 0.0018 to 0.0054 99.8% (n=3,881), 0.0054 to 0.039 98.4% (n=3,878). All together: 3,555 excess failures = 24% of all prediction failures.
- Failure type: mostly **wrong_class** (63% of failures here vs 36% overall, 1.7× over-represented). Mix: wrong_class 63%, background 24%, mislocalized 12%, duplicate 1%.
- Confounding: **holds** within every other attribute (12 tested; the strongest, category, still leaves +21.9% points of the +31.1% gap).
- What to check: Low-confidence region: this is a threshold choice, see *Calibration and threshold* below.
- Phoenix: `name == 'prediction' and metadata['score'] >= 0.001 and metadata['score'] <= 0.0018 and annotations['outcome'].label == 'wrong_class'`

### 8. prediction: category = Platelets
**91.1% wrong** vs 71.2% for the rest (1.28×, 95% CI 90.2%–91.9%). 4,021 predictions, 3,662 failures, **797 more than expected**.
- Other significantly worse values of `category`: WBC 88.4% (n=3,197). All together: 1,296 excess failures = 9% of all prediction failures.
- Failure type: mostly **wrong_class** (63% of failures here vs 36% overall, 1.7× over-represented). Mix: wrong_class 63%, background 27%, mislocalized 10%, duplicate 1%.
- Confounding: **holds** within every other attribute (11 tested; the strongest, area_px (quintile), still leaves +15.5% points of the +19.8% gap).
- What to check: Right place, wrong class. See the confusion pairs; ambiguous classes may need clearer labelling guidelines, merging, or more examples.
- Phoenix: `name == 'prediction' and metadata['category'] == 'Platelets' and annotations['outcome'].label == 'wrong_class'`

Smaller problems (also significant, not detailed): prediction: size = medium (82.1% vs 70.8%, 7%); prediction: area_px (quintile) = 51.8 to 5832 (89.6% vs 71.7%, 6%); prediction: aspect_ratio (quintile) = 1.238 to 7.87 (85.3% vs 72.8%, 5%); prediction: n_objects (quintile) = 1 to 10 (85.9% vs 72.7%, 4%); prediction: n_objects_bucket = 8-15 (76.2% vs 74.0%, 4%); prediction: position = edge (81.8% vs 73.2%, 3%); prediction: n_preds (quintile) = 69 to 100 (80.1% vs 74.1%, 2%); prediction: aspect = tall (99.4% vs 74.9%, 1%).

## 3. Looks like a problem, but is explained by something else

These slices fail more often overall, but the gap mostly disappears once you compare like with like.

- **prediction: truncated = True** (78.8% vs 74.6%): within *aspect_ratio (quintile)* the gap falls from +4.2% to -1.9% points.
- **prediction: n_classes = 2** (76.0% vs 74.8%): within *n_objects (quintile)* the gap falls from +1.2% to -2.1% points.

## 4. Calibration and threshold

`pred_units`: share correct by `score`.

| score | n | correct | most common error |
|---|---|---|---|
| (-0.001, 0.1] | 12945 | 1.3% | wrong_class (42%) |
| (0.1, 0.2] | 878 | 19.9% | background (57%) |
| (0.2, 0.3] | 551 | 29.0% | background (67%) |
| (0.3, 0.4] | 470 | 40.0% | background (64%) |
| (0.4, 0.5] | 409 | 53.3% | background (69%) |
| (0.5, 0.6] | 382 | 63.6% | background (65%) |
| (0.6, 0.7] | 422 | 73.7% | background (68%) |
| (0.7, 0.8] | 456 | 82.2% | background (75%) |
| (0.8, 0.9] | 732 | 90.0% | background (86%) |
| (0.9, 1.0] | 2399 | 98.2% | background (82%) |

Threshold trade-off (recall = share of ground truth still found):

| threshold | kept | precision | recall | f1 |
|---|---|---|---|---|
| 0.05 | 7533 | 63.3% | 97.5% | 76.7% |
| 0.1 | 6699 | 69.9% | 95.9% | 80.9% |
| 0.15 | 6218 | 73.9% | 94.1% | 82.8% |
| 0.2 | 5822 | 77.4% | 92.3% | 84.2% |
| 0.25 | 5535 | 80.2% | 90.9% | 85.2% |
| 0.3 | 5270 | 82.5% | 89.0% | 85.6% |
| 0.35 | 5009 | 84.9% | 87.0% | 85.9% |
| 0.4 | 4800 | 86.7% | 85.2% | 85.9% |
| 0.45 | 4586 | 88.4% | 83.0% | 85.6% |
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

Best F1 at score ≥ **0.35** (precision 84.9%, recall 87.0%). Whether that is the right point depends on whether misses or false alarms cost more.

## 5. Class confusions

5,356 pred_units rows are in the right place with the wrong class. Most common pairs (`category` → `gt_label`):

| category | gt_label | count |
|---|---|---|
| WBC | RBC | 2029 |
| Platelets | RBC | 1557 |
| Platelets | WBC | 746 |
| RBC | WBC | 373 |
| WBC | Platelets | 333 |
| RBC | Platelets | 318 |

Both directions combined: RBC ↔ WBC (2402), Platelets ↔ RBC (1875), Platelets ↔ WBC (1079).

Pairs that go both ways are often ambiguous in the labels too; check them with the label pipeline.

## 6. Possible label issues

36 of 5,733 detections on unlabelled regions score ≥ 0.9. Confident detections with nothing labelled underneath are often objects missing from the labels, not model errors.

| category | count |
|---|---|
| RBC | 30 |
| Platelets | 5 |
| WBC | 1 |

Phoenix: `name == 'prediction' and annotations['outcome'].label == 'background' and metadata['score'] >= 0.9`

## 7. Localization sensitivity

11 gt_units rows have a box nearby that overlaps too little. Their best IoU:

| IoU | count |
|---|---|
| (0.0, 0.1] | 0 |
| (0.1, 0.2] | 1 |
| (0.2, 0.3] | 0 |
| (0.3, 0.4] | 1 |
| (0.4, 0.5] | 9 |

Loosening the match would raise the correct rate from 99.3% to 99.4% at IoU 0.4, or 99.4% at IoU 0.3. Small gain: when boxes are off, they are well off.

## 8. Where failures concentrate

- `gt_units`: the worst 10% of images hold **100%** of failures; 0 of 364 images have none of their objects correct. Worst images: BloodImage_00037 (3), BloodImage_00295 (2), BloodImage_00029 (2), BloodImage_00038 (2), BloodImage_00010 (2).
- `pred_units`: the worst 10% of images hold **16%** of failures; 0 of 364 images have none of their predictions correct. Worst images: BloodImage_00063 (90), BloodImage_00067 (83), BloodImage_00091 (81), BloodImage_00050 (79), BloodImage_00030 (76).

## 9. Every attribute: its worst value

The single worst value of each attribute, significant or not (p_adj = Benjamini-Hochberg corrected).

**`gt_object`** (overall 0.7% not found)

| attribute | worst | n | fail | rest | lift | excess | significant |
|---|---|---|---|---|---|---|---|
| aspect | squarish | 4,859 | 0.7% | 0.0% | inf | 34 | no |
| overlap (quintile) | 0.109 to 0.933 | 977 | 3.3% | 0.1% | 64.02 | 32 | yes |
| category | RBC | 4,153 | 0.8% | 0.3% | 2.82 | 21 | no |
| size | medium | 1,524 | 1.5% | 0.3% | 4.61 | 18 | yes |
| n_objects_bucket | 16+ | 1,932 | 1.2% | 0.4% | 3.20 | 16 | yes |
| n_objects (quintile) | 18 to 30 | 820 | 2.0% | 0.4% | 4.41 | 12 | yes |
| n_preds (quintile) | 66 to 100 | 937 | 1.6% | 0.5% | 3.33 | 10 | yes |
| rel_area (quintile) | 0.00142 to 0.02635 | 981 | 1.5% | 0.5% | 3.14 | 10 | yes |
| area_px (quintile) | 437 to 8096 | 981 | 1.5% | 0.5% | 3.14 | 10 | yes |
| truncated | False | 3,704 | 0.8% | 0.5% | 1.49 | 9 | no |
| position | center | 968 | 1.1% | 0.6% | 1.94 | 5 | no |
| center_dist (quintile) | 0.011 to 0.333 | 978 | 1.1% | 0.6% | 1.91 | 5 | no |
| n_classes | 3 | 2,854 | 0.8% | 0.6% | 1.31 | 5 | no |
| aspect_ratio (quintile) | 0.333 to 0.924 | 982 | 1.0% | 0.6% | 1.66 | 4 | no |

**`prediction`** (overall 75.3% wrong)

| attribute | worst | n | fail | rest | lift | excess | significant |
|---|---|---|---|---|---|---|---|
| score_bin | 0.5-0.6 | 15,635 | 92.6% | 7.7% | 12.02 | 13,278 | yes |
| score (quintile) | 0.001 to 0.0018 | 4,028 | 100.0% | 68.9% | 1.45 | 1,252 | yes |
| category | Platelets | 4,021 | 91.1% | 71.2% | 1.28 | 797 | yes |
| rel_area (quintile) | 0.00017 to 0.01898 | 3,930 | 89.6% | 71.7% | 1.25 | 705 | yes |
| area_px (quintile) | 51.8 to 5832 | 3,929 | 89.6% | 71.7% | 1.25 | 704 | yes |
| n_objects (quintile) | 1 to 10 | 3,936 | 85.9% | 72.7% | 1.18 | 520 | yes |
| aspect_ratio (quintile) | 1.238 to 7.87 | 3,915 | 85.3% | 72.8% | 1.17 | 487 | yes |
| position | edge | 4,781 | 81.8% | 73.2% | 1.12 | 408 | yes |
| center_dist (quintile) | 0.681 to 0.952 | 3,887 | 82.2% | 73.6% | 1.12 | 335 | yes |
| n_preds (quintile) | 69 to 100 | 3,839 | 80.1% | 74.1% | 1.08 | 230 | yes |
| truncated | True | 3,228 | 78.8% | 74.6% | 1.06 | 134 | yes |
| size | small | 535 | 95.9% | 74.7% | 1.28 | 113 | yes |
| n_classes | 1 | 510 | 91.4% | 74.9% | 1.22 | 84 | yes |
| aspect | tall | 342 | 99.4% | 74.9% | 1.33 | 84 | yes |
| n_objects_bucket | 1 | 310 | 98.1% | 74.9% | 1.31 | 72 | yes |

## 10. Caveats

- Everything here is correlation over this evaluation set. A slice tells you where to look, not why.
- Confounding is checked one attribute at a time; two attributes together can still explain a slice.
- Slices smaller than the minimum size are not reported; small slices have wide confidence intervals.
