# Black boxes

Each folder here is one annotation-audit method. `run_audit.py` finds them by their
`blackbox_config.json`, runs each one, checks its output and hands it to `ensemble_audit.py`.
Neither of those scripts knows how any method works, so **adding a method never means
editing them**: you add a folder.

```
python run_audit.py --list-methods          # what's here, and how each one runs
```

## What a black box is

A script that judges how suspicious each image's **annotations** are. It runs as its
own process (so it can have its own dependencies), and writes two files into `--out-dir`:

| file | what's in it |
|---|---|
| `scores.csv` | one row per image: `file_name`, `file_path`, one or more `s_<signal>` columns, `flagged`, `reason` |
| `meta.json` | `name`, `method`, `group`, `signals`, optional `weights`, `params` |

`write_blackbox()` in `blackbox_common.py` writes both; use it. The full contract is
in that file's docstring.

## blackbox_config.json

```json
{
  "description": "one line for --list-methods",
  "script": "blackbox_mymethod.py",
  "predictions": "none",
  "accepts": ["work_dir"],
  "prefix": "mymethod",
  "weight": 1,
  "to_run": true
}
```

| key | required | meaning |
|---|---|---|
| `script` | yes | path to the script, relative to this `blackbox_config.json` |
| `predictions` | yes | `"per_model"`: runs once per `--predictions` file and gets `--predictions <file>`; output `scores/<prefix>_<tag>`. `"none"`: runs once and uses only the labels and images; output `scores/<prefix>` |
| `accepts` | no | which optional standard arguments the script understands (below). Only these are passed |
| `prefix` | no | output folder name, default: this folder's name |
| `weight` | no | default weight of the method in the ensemble (default 1). `run_audit.py --weights` overrides it |
| `to_run` | no | `false`: only runs when named in `--methods` (good for experiments). Default `true` |
| `description` | no | shown by `--list-methods` |

### Arguments your script gets

Always: `--labels`, `--images-dir`, `--out-dir`. With `"per_model"`: `--predictions`.
And, only if listed in `accepts`:

| `accepts` entry | argument | use it for |
|---|---|---|
| `work_dir` | `--work-dir <root>/work/<name>` | intermediate files you want to keep / reuse |
| `evaluate_planted` | `--evaluate-planted` | test sets: report how many planted `*_wrong` / `*_missing` images you catch |
| `audit_name` | `--audit-name <name>` | state that lives outside the audit folder (a database, cache, index): name it after this so every audit has its own |

Plus whatever the user passes with `run_audit.py --box-args <method>="..."`, so give
your script its own options (thresholds etc.) with sensible defaults.

The script runs from the folder `run_audit.py` was started in (paths in the arguments are
relative to it), with `cv_annotation/` on `PYTHONPATH`, so `from blackbox_common import ...`
works. If you also want to run it by hand, copy the `sys.path.insert` line from
`geometry/blackbox_geometry.py`.

Because the working directory isn't your folder, reach files that live next to your script
(helper scripts, model weights, ...) through the script's own location, not by bare name:
`os.path.join(os.path.dirname(os.path.abspath(__file__)), "helper.py")`, as
`cleanlab/blackbox_cleanlab.py` does for `cleanlab_data_audit.py`.

## Adding a method, step by step

The quickest start is the template: `cp -r blackbox_template blackboxes/<method>`, rename
the script and fill in its `score_image()` (see the top of `blackbox_template/blackbox_template.py`).
It does steps 1–3 for you except your own scoring code. To write one from scratch:

1. **Make a folder** `blackboxes/<method>/`. The folder name is the method's name in
   `--methods`.
2. **Write the script.** `geometry/blackbox_geometry.py` is a complete example of a
   `"none"` box; `predictions/blackbox_predictions.py` of a `"per_model"` one that wraps another tool.
   For every image in `image_paths(labels, images_dir)`, work out your signals, then call
   `write_blackbox(out_dir, rows, name=..., method=..., group=..., signals=..., params=...)`.
3. **Write `blackbox_config.json`** next to it.
4. **Check it's found**: `python run_audit.py --list-methods`.
5. **Test it alone on a planted test set**, e.g.
   `python run_audit.py --labels ../DATA/labels_sub500_v2.json --images-dir ../DATA/data_sub500_v2 --methods <method> --predictions "" --evaluate-planted --root /tmp/try_<method>`
   (drop `--predictions ""` for a per_model box). `ensemble/evaluation.csv` says how many
   planted errors it ranks near the top and its AUC per kind (wrong / missing).
6. **Test it with the others**: run the full audit with and without it and compare
   `evaluation.csv`. The `rrf without <group>` rows show what each group adds. A method that
   finds the same images as the rest (see `method_correlation.csv`) adds little.
7. Set `"to_run": false` until you're happy with it; then remove it.

## Things to get right

- **Higher = more suspicious**, for every `s_` column. Scale doesn't matter (each signal
  becomes a percentile rank), direction does. If your tool gives a quality score, write
  `1 - quality`.
- **Blank means "no opinion"**, `0` means "looked, found nothing". Use blank for images
  your method can't judge (no boxes, too few examples of a class, ...): a blank doesn't
  count for or against the image, while a 0 ranks it as clean.
- **One row per image in `--labels`**, keyed by `file_name` exactly as it is there. Missing
  rows are treated as no opinion (and warned about); extra rows are ignored.
- **Pick `group` honestly.** It's the independent evidence the method relies on. Methods
  sharing a group are averaged into one vote, because they're correlated. Built on one
  model's predictions: the predictions tag (`predictions_tag(path)`). Uses no model: its
  own name. Getting this wrong double-counts evidence.
- **`name` must be unique** across the black boxes of one audit: for per_model boxes
  include the tag (`<method>_<tag>`).
- **Count signals tie a lot.** Most images have 0 of a rare thing, and tied values share a
  middle percentile. That's fine, but a signal that's 0 almost everywhere says little
  except about the few images where it isn't.
- **`flagged` and `reason`** are your method's own verdict and its worst finding. The
  ensemble shows them on the review list, so make `reason` readable on its own
  ("degenerate: annotated skis box 19.9x2.0px").
- **Record your settings in `params`**, so a scores folder says how it was made.
- **Weights.** Per-signal default weights go in `write_blackbox(..., weights=...)` (0 drops
  a signal); the method's weight goes in `blackbox_config.json`. For a box alone in its group, that
  weight is the group's vote.
- **`--reuse`** in `run_audit.py` skips a box whose `scores/<name>/meta.json` exists, without
  looking at whether its inputs changed. Rerun without `--reuse` after changing the labels,
  the predictions or your script.

`run_audit.py` checks every box's output before the ensemble reads it (`check_blackbox()` in
`blackbox_common.py`) and stops with a list of what's wrong if it breaks the contract.
