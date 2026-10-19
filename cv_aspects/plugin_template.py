"""
plugin_template.py -- bring your own "split by aspect" black box.

HOW TO USE
    1. Copy this file anywhere (e.g. my_weather.py). Do not edit it inside cv_aspects.
    2. Fill in the three places marked  >>> FILL IN <<<  below.
    3. Split the images with your aspect:
           python split_by_aspect.py --images ../DATA/bccd_images --plugin my_weather.py --aspects plugins
       (use --aspects all to also run the built-in aspects, or --aspects my_aspect,brightness to pick)
    4. Evaluate any model on the groups, exactly as for the built-in aspects:
           python evaluate_by_aspect.py --preds ... --labels ... --tags aspects_out/tags.csv

WHAT YOUR CODE RECEIVES
    image_paths  list of full paths, one per image, in a fixed order
    labels       None, or {file_name: [{"bbox": [x, y, w, h], "category": "RBC", "iscrowd": 0}, ...]}
                 (only when --labels was given; set NEEDS_LABELS = True to require it)
    options      dict of your own settings from the --config JSON under your NAME, e.g.
                 {"my_aspect": {"model": "C:/models/weather.pt", "threshold": 0.5}}
                 -> options == {"model": "C:/models/weather.pt", "threshold": 0.5}

WHAT YOUR CODE RETURNS - one value per image, same order as image_paths (or a dict keyed by path)
    KIND = "categorical"  a label per image, e.g. "sunny" / "rainy". Each label is a group.
    KIND = "numeric"      a number per image, e.g. 0.73. Binned into quantile groups, or fixed edges
                          from the config: {"my_aspect": {"edges": [0.3, 0.7], "names": ["low", "mid", "high"]}}
    KIND = "embedding"    a vector per image (shape N x D). Clustered with KMeans into
                          options["k"] groups (default 5), named c0 (largest) .. c<k-1>.
    Use None for "could not decide"; those images go into the group "n/a".

Nothing here needs to import from cv_aspects. Your file can import torch, your own model code, etc.
"""

# ------------------------------------------------------------------------------------------------
# >>> FILL IN 1 <<<  describe your aspect
# ------------------------------------------------------------------------------------------------
NAME = "my_aspect"            # column name in tags.csv and in the reports; must be unique
KIND = "categorical"          # "categorical" | "numeric" | "embedding"
DESCRIPTION = "what this aspect measures"
NEEDS_LABELS = False          # True -> skipped unless --labels is given


# ------------------------------------------------------------------------------------------------
# >>> FILL IN 2 <<<  (optional) load your model / resources once, before any image is processed
# ------------------------------------------------------------------------------------------------
def setup(options):
    """Return anything you need later (a model, a lookup table, ...). Called once."""
    # e.g.  import torch; return torch.load(options["model"])
    return None


# ------------------------------------------------------------------------------------------------
# >>> FILL IN 3 <<<  decide the value for ONE image
# ------------------------------------------------------------------------------------------------
def value_for_image(path, boxes, state, options):
    """
    path     full path to the image file
    boxes    this image's GT boxes (list of dicts, see above), or None without --labels
    state    whatever setup() returned
    return   a label (categorical), a number (numeric) or a 1-D vector (embedding); None = unknown
    """
    raise NotImplementedError("fill in value_for_image() in your copy of plugin_template.py")


# ------------------------------------------------------------------------------------------------
# Entry point called by split_by_aspect.py. Usually no need to change it; replace the loop if you
# want to batch images through a model (it must still return one value per image, in order).
# ------------------------------------------------------------------------------------------------
def compute(image_paths, labels=None, options=None):
    import os

    options = options or {}
    state = setup(options)
    values = []
    for path in image_paths:
        boxes = labels.get(os.path.basename(path)) if labels is not None else None
        values.append(value_for_image(path, boxes, state, options))
    return values
