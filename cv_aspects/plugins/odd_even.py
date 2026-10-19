"""
Example plugin made from plugin_template.py: is the number at the end of the file name odd or even?

    BloodImage_00007.jpg          -> odd
    BloodImage_00010.jpg          -> even
    BloodImage_00368_missing.jpg  -> neither   (name does not end in a digit)

    python split_by_aspect.py --images ../DATA/bccdp_images --plugin plugins/odd_even.py --aspects plugins
"""

import os
import re

# >>> FILL IN 1 <<<
NAME = "odd_even"
KIND = "categorical"
DESCRIPTION = "whether the file name (without extension) ends in an odd or even number"
NEEDS_LABELS = False


# >>> FILL IN 2 <<<  nothing to load for this one
def setup(options):
    return None


# >>> FILL IN 3 <<<
def value_for_image(path, boxes, state, options):
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"(\d+)$", stem)
    if m is None:
        return "neither"
    return "odd" if int(m.group(1)) % 2 else "even"


# unchanged from the template
def compute(image_paths, labels=None, options=None):
    options = options or {}
    state = setup(options)
    values = []
    for path in image_paths:
        boxes = labels.get(os.path.basename(path)) if labels is not None else None
        values.append(value_for_image(path, boxes, state, options))
    return values
