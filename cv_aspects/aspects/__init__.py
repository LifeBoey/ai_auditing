"""
Registry of aspects: the built-ins below plus any plugin files given with --plugin.
Bring your own aspect without editing this package: copy plugin_template.py, fill it in, and run
split_by_aspect.py --plugin your_file.py.
"""

from .base import Aspect, Context
from .clip import ClipClusterAspect
from .gt import gt_aspects
from .pixel import PIXEL_ASPECTS
from .plugin import load_plugin


def builtin(ctx):
    classes = PIXEL_ASPECTS + gt_aspects(ctx) + [ClipClusterAspect]
    return {c.name: c() for c in classes}


def available(ctx, plugins=()):
    """name -> Aspect instance: built-ins plus the plugin aspects (already loaded)."""
    out = builtin(ctx)
    for p in plugins:
        if p.name in out:
            raise SystemExit(f"plugin {p.path} uses the name {p.name!r}, which is already taken "
                             f"(built-in or another plugin); set NAME = '...' in the plugin")
        out[p.name] = p
    return out
