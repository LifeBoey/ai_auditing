"""
Turn raw aspect values into groups.

Per-aspect config (all optional):
    {"brightness": {"edges": [120, 160], "names": ["dark", "normal", "bright"]},   # fixed thresholds
     "n_gt":       {"n_bins": 3},                                                  # quantiles
     "default_n_bins": 4}
Fixed edges are left-closed: [-inf, e0), [e0, e1), ..., [e_last, inf).
"""

import numpy as np
import pandas as pd

MISSING = "n/a"


def _fmt(v):
    return f"{v:.3g}"


def _assign(v, edges, names):
    idx = np.searchsorted(edges, v.to_numpy(), side="right")
    lab = np.array(names, dtype=object)[idx]
    lab[v.isna().to_numpy()] = MISSING
    return pd.Series(lab, index=v.index)


def bin_numeric(v, cfg, default_n_bins=4):
    v = pd.to_numeric(v, errors="coerce")
    ok = v.dropna()
    if "edges" in cfg:
        edges = sorted(float(e) for e in cfg["edges"])
        method = "fixed"
    else:
        n = int(cfg.get("n_bins", default_n_bins))
        edges = sorted(set(np.quantile(ok, np.linspace(0, 1, n + 1)[1:-1]).tolist())) if len(ok) else []
        # an edge equal to the minimum would leave the first bin empty
        edges = [e for e in edges if e > ok.min()] if len(ok) else []
        if not edges and ok.nunique() > 1:
            # most values tie at the minimum (e.g. 0): split "= min" from "> min"
            edges = [float(np.sort(ok.unique())[1])]
        method = f"quantile({n})"
    if "names" in cfg:
        names = list(cfg["names"])
        if len(names) != len(edges) + 1:
            raise SystemExit(f"config: {len(edges)} edges need {len(edges) + 1} names, got {names}")
    else:
        lo = [ok.min() if len(ok) else -np.inf] + edges
        hi = edges + [ok.max() if len(ok) else np.inf]
        names = [f"{i + 1}: [{_fmt(a)}, {_fmt(b)}{']' if i == len(edges) else ')'}"
                 for i, (a, b) in enumerate(zip(lo, hi))]
    groups = _assign(v, edges, names)
    order = names + ([MISSING] if (groups == MISSING).any() else [])
    return groups, dict(method=method, edges=edges, groups=order)


def bin_categorical(v):
    groups = v.astype(object).where(v.notna(), MISSING).astype(str)
    order = sorted(groups.unique(), key=lambda g: (g == MISSING, _natural(g)))
    return groups, dict(method="categorical", edges=None, groups=order)


def _natural(g):
    try:
        return (0, float(g), "")
    except ValueError:
        return (1, 0.0, g)
