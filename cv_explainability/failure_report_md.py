"""
failure_report_md.py -- turn the per-unit CSVs (from phoenix_slices.py, or any similar run) into a
Markdown report of the main problems to look out for.

Generic input: every CSV in --dir that has a `correct` column (True/False) is a "unit" table, one row
per thing that can be right or wrong (a ground-truth object, a prediction, an image, ...). Optional
columns make extra sections appear:
    outcome        failure type (missed, mislocalized, background, ...)  -> failure-mode breakdown
    image_id       which image a row belongs to                          -> where failures concentrate
    score          model confidence                                      -> calibration + threshold sweep
    iou            overlap with the closest match                        -> localization sensitivity
    *_label        the "other side's" class (pred_label / gt_label)      -> confusion pairs
Every other low-cardinality column, or numeric column (cut into quintiles), is a slice attribute.

What the report does
    1. Overview: size, accuracy and outcome mix of each unit table (the same numbers Phoenix shows
       as the `correct` / `outcome` annotation summaries).
    2. Slices: every attribute value tested against the rest (one-sided z-test, Benjamini-Hochberg),
       ranked by EXCESS failures (failures beyond what the overall rate predicts), near-duplicates merged.
    3. Confounding: each top slice is re-measured inside the strata of every other attribute.
       If the gap disappears (e.g. "indoor" only fails because indoor objects are small), it is moved
       to "looks like a problem, but ...".
    4. Problem cards: rate, impact, dominant failure type, confounder verdict, what to check, and a
       Phoenix span filter to see the examples.
    5. Detection extras when the columns exist: calibration, threshold trade-off, class confusions,
       confident detections on unlabelled regions, IoU sensitivity, failure concentration.

Run:  python failure_report_md.py [--dir phoenix_slices] [--out <dir>/failure_report.md] [--top 8]
"""

import argparse
import glob
import math
import os
import re
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SPAN_NAMES = {"gt_units": "gt_object", "pred_units": "prediction", "images": "image"}
UNIT_WORDS = {"gt_units": ("object", "not found"), "pred_units": ("prediction", "wrong")}

SKIP_COLS = {"correct", "outcome", "explain", "explanation", "iou", "filepath", "file_name", "path",
             "x", "y", "w", "h", "x1", "y1", "x2", "y2", "pred_label", "pred_score", "gt_label",
             "recall", "precision", "n_detected", "n_tp"}       # identifiers, geometry, outcome leaks
SKIP_RE = re.compile(r"(^id$|_id$|_idx$|^idx$|_conf$)")
MAX_LEVELS = 100           # string columns with more distinct values are treated as free text
MIN_V_SAME = 0.6           # Cramér's V above which two attributes measure "the same thing"
# produced by the model itself: sliced on, but never used to "explain away" a data slice
MODEL_OUTPUTS = {"score", "confidence", "score_bin", "n_preds"}

ADVICE = {
    "missed": "The model draws no box here at all. Check input resolution / small-object capacity, and "
              "whether lower-scored boxes exist (re-run with a lower score threshold to split "
              "'never proposed' from 'scored too low').",
    "mislocalized": "A box is nearby but misplaced (IoU too low). Look at box quality: occlusion, "
                    "truncation, unusual aspect ratios, or inconsistent box conventions in the labels.",
    "confused": "The box is in the right place with the wrong class. See the confusion pairs; ambiguous "
                "classes may need clearer labelling guidelines, merging, or more examples.",
    "wrong_class": "Right place, wrong class. See the confusion pairs; ambiguous classes may need clearer "
                   "labelling guidelines, merging, or more examples.",
    "claimed_by_neighbour": "Adjacent objects compete for one box: crowding / overlap handling (NMS).",
    "background": "Boxes where nothing is labelled. Check for unlabelled objects first (label pipeline), "
                  "then treat the rest as false alarms (threshold, hard negatives).",
    "duplicate": "Several boxes on one object: tune NMS (IoU threshold, class-agnostic NMS).",
}


# ============================================================
# loading + attribute discovery
# ============================================================
def load_units(d):
    units = {}
    for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
        head = pd.read_csv(f, nrows=0).columns
        if "correct" not in head:
            continue
        df = pd.read_csv(f, low_memory=False, dtype={c: str for c in head if c == "image_id"})
        df["correct"] = df["correct"].astype(str).str.lower().isin(["true", "1", "1.0"])
        units[os.path.splitext(os.path.basename(f))[0]] = df
    return units


def binned_attributes(df, skip):
    """{attr: (labels Series[str], kind, edges)}; kind in str|bool|num|quint."""
    out = {}
    for c in df.columns:
        if c in SKIP_COLS or c in skip or SKIP_RE.search(c):
            continue
        s = df[c]
        if s.notna().sum() == 0 or s.nunique(dropna=True) < 2:
            continue
        if s.dtype == bool:
            out[c] = (s.astype(str), "bool", None)
        elif pd.api.types.is_numeric_dtype(s):
            if s.nunique() <= 10:
                out[c] = (s.map(lambda v: "nan" if pd.isna(v) else f"{v:g}"), "num", None)
            else:
                q, edges = pd.qcut(s, 5, duplicates="drop", retbins=True, labels=False)
                lab = q.map(lambda i: "nan" if pd.isna(i) else f"{edges[int(i)]:.4g} to {edges[int(i) + 1]:.4g}")
                out[f"{c} (quintile)"] = (lab, "quint", (c, edges, q))
        elif s.nunique() <= MAX_LEVELS:
            out[c] = (s.astype(str), "str", None)
    return out


# ============================================================
# statistics
# ============================================================
def _wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def _bh(p):
    p = np.asarray(p, float)
    if not len(p):
        return p
    o = np.argsort(p)
    adj = np.empty(len(p))
    adj[o] = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    return np.minimum(adj, 1)


def slice_stats(fail, attrs, min_n):
    N, F = len(fail), int(fail.sum())
    rows = []
    for a, (lab, kind, _) in attrs.items():
        for v, idx in lab.groupby(lab).groups.items():
            n = len(idx)
            if n < min_n or v == "nan" or n == N:
                continue
            k = int(fail.loc[idx].sum())
            rn, rk = N - n, F - k
            rate, rest = k / n, rk / rn
            pool = F / N
            se = math.sqrt(pool * (1 - pool) * (1 / n + 1 / rn)) if 0 < pool < 1 else 0
            z = (rate - rest) / se if se else 0
            lo, hi = _wilson(k, n)
            rows.append(dict(attribute=a, value=v, kind=kind, n=n, n_fail=k, rate=rate, rest=rest, ci_low=lo,
                             ci_high=hi, lift=rate / rest if rest else np.inf, excess=k - n * rest,
                             p=0.5 * math.erfc(z / math.sqrt(2))))
    t = pd.DataFrame(rows)
    if len(t):
        t["p_adj"] = _bh(t["p"])
        t["excess_share"] = t["excess"] / max(F, 1)
    return t


def cramers_v(a, b):
    tab = pd.crosstab(a, b).to_numpy(float)
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return 1.0
    e = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / tab.sum()
    return math.sqrt(((tab - e) ** 2 / e).sum() / tab.sum() / (min(tab.shape) - 1))


def base_col(attr):
    return attr.replace(" (quintile)", "")


def confound(fail, mask, attr, attrs, min_cell=10):
    """Re-measure the slice's gap inside every other attribute's strata. -> (retained, by, adj, tested)"""
    raw = fail[mask].mean() - fail[~mask].mean()
    best, tested = (1.0, None, None), 0
    own = attrs[attr][0]
    for b, (lab, _, _) in attrs.items():
        if b == attr or base_col(b) in MODEL_OUTPUTS:
            continue
        if cramers_v(mask, lab) > MIN_V_SAME or cramers_v(own, lab) > MIN_V_SAME:   # same thing twice
            continue
        g = pd.DataFrame(dict(f=fail.to_numpy(), s=mask.to_numpy(), b=lab.to_numpy()))
        g = g[g.b != "nan"].groupby(["b", "s"])["f"].agg(["sum", "count"]).unstack("s")
        if ("count", True) not in g or ("count", False) not in g:
            continue
        g = g[(g[("count", True)] >= min_cell) & (g[("count", False)] >= min_cell)]
        ns = g[("count", True)]
        if ns.sum() < 0.5 * mask.sum():                         # too little of the slice comparable
            continue
        gap = (g[("sum", True)] / ns - g[("sum", False)] / g[("count", False)])
        adj = float((gap * ns).sum() / ns.sum())
        tested += 1
        r = adj / raw if raw else 1.0
        if r < best[0]:
            best = (r, b, adj)
    return best[0], best[1], best[2], tested


# ============================================================
# report helpers
# ============================================================
pct = lambda x: "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.1%}"


def md_table(df, floatfmt=None):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(out)


def phoenix_filter(span, attr, value, kind, attrs, extra=""):
    lab, _, meta = attrs[attr]
    if kind == "quint":
        col, edges, q = meta
        i = int(q[lab == value].dropna().iloc[0])
        lo_op = ">=" if i == 0 else ">"
        f = f"metadata['{col}'] {lo_op} {edges[i]:.6g} and metadata['{col}'] <= {edges[i + 1]:.6g}"
    elif kind == "bool":
        f = f"metadata['{attr}'] == {value}"
    elif kind == "num":
        f = f"metadata['{attr}'] == {value}"
    else:
        f = f"metadata['{attr}'] == '{value}'"
    return f"name == '{span}' and {f}{extra}"


def outcome_mix(df, mask=None):
    d = df if mask is None else df[mask]
    f = d[~d["correct"]]
    return f["outcome"].value_counts(normalize=True) if "outcome" in d and len(f) else pd.Series(dtype=float)


def dominant_mode(mix, base_mix, min_share=0.2):
    """The failure type most over-represented here (among those with a real share), else the commonest."""
    if not len(mix):
        return None, None, None
    cand = [(mix[k] / base_mix.get(k, np.nan), k) for k in mix.index if mix[k] >= min_share and base_mix.get(k, 0) > 0]
    ratio, k = max(cand) if cand else (np.nan, mix.index[0])
    if not ratio > 1.15:
        k = mix.index[0]
    return k, mix[k], base_mix.get(k, 0)


# ============================================================
# analysis per unit table
# ============================================================
def analyse(name, df, min_n, top, skip):
    fail = ~df["correct"]
    attrs = binned_attributes(df, skip)
    t = slice_stats(fail, attrs, min_n)
    res = dict(name=name, df=df, attrs=attrs, table=t, base=fail.mean(), span=SPAN_NAMES.get(name, name),
               words=UNIT_WORDS.get(name, ("row", "incorrect")), cards=[], explained=[])
    if not len(t):
        return res
    worse = t[(t.p_adj < 0.05) & (t.lift > 1)].sort_values("excess", ascending=False)
    base_mix = outcome_mix(df)
    # one card per attribute: headline = its value with the most excess failures, the rest listed with it;
    # attributes are ranked by the excess of all their significantly worse values together
    per_attr = sorted(worse.groupby("attribute"), key=lambda g: -g[1].excess.sum())
    kept, masks = [], []
    for attr, g in per_attr:
        r = next(g.itertuples())
        m = attrs[attr][0] == r.value
        dup = next((k for k, km in zip(kept, masks) if (m & km).sum() / (m | km).sum() >= 0.6), None)
        if dup is not None:
            dup.setdefault("also", []).append(f"{attr} = {r.value}")
            continue
        retained, by, adj, tested = confound(fail, m, attr, attrs)
        mix = outcome_mix(df, m)
        mode, mode_share, mode_base = dominant_mode(mix, base_mix)
        card = dict(r._asdict(), mask=m, retained=retained, by=by, adj=adj, tested=tested, mode=mode,
                    mode_share=mode_share, mode_base=mode_base, mix=mix,
                    total_excess=g.excess.sum(), total_share=g.excess.sum() / max(int(fail.sum()), 1),
                    others=g.iloc[1:].sort_values("lift", ascending=False))
        kept.append(card)
        masks.append(m)
        if len(kept) >= 3 * top:
            break
    for c in kept:
        (res["explained"] if c["retained"] < 1 / 3 else res["cards"]).append(c)
    return res


def calibration(units):
    """Find a unit with a score column; precision by score and a threshold sweep."""
    for name, df in units.items():
        sc = next((c for c in ("score", "confidence") if c in df and pd.api.types.is_numeric_dtype(df[c])), None)
        if sc is None:
            continue
        lo = math.floor(df[sc].min() * 10) / 10
        bins = np.round(np.arange(lo, 1.0001, 0.1), 2)
        b = pd.cut(df[sc], bins, include_lowest=True)
        tab = df.groupby(b, observed=True).agg(n=("correct", "size"), correct=("correct", "mean"))
        mix = df[~df.correct].groupby(b, observed=True)["outcome"].agg(
            lambda s: f"{s.value_counts().index[0]} ({s.value_counts(normalize=True).iloc[0]:.0%})") if "outcome" in df else None
        tab = tab.reset_index().rename(columns={sc: "score"})
        tab["correct"] = tab["correct"].map(pct)
        if mix is not None:
            tab["most common error"] = mix.reindex(tab["score"]).to_numpy()
        tab["score"] = tab["score"].astype(str)
        # threshold sweep; recall only if another unit's correct count equals this one's (one-to-one matching)
        other = next((d for n, d in units.items() if n != name and int(d.correct.sum()) == int(df.correct.sum())
                      and "score" not in d), None)
        sweep = []
        for th in np.round(np.arange(max(lo, 0.05), 0.96, 0.05), 2):
            k = df[df[sc] >= th]
            if not len(k):
                continue
            p_, tp = k.correct.mean(), int(k.correct.sum())
            row = dict(threshold=th, kept=len(k), precision=p_)
            if other is not None:
                rec = tp / len(other)
                row.update(recall=rec, f1=2 * p_ * rec / (p_ + rec) if p_ + rec else 0)
            sweep.append(row)
        return name, sc, tab, pd.DataFrame(sweep), other is not None, df[sc].min()
    return None


def confusions(units, top=15):
    for name, df in units.items():
        if "outcome" not in df:
            continue
        other = next((c for c in df.columns if c.endswith("_label")), None)
        own = next((c for c in ("category", "label", "class") if c in df), None)
        conf = df[df.outcome.isin(["confused", "wrong_class"])]
        if other and own and len(conf):
            pairs = conf.groupby([own, other]).size().sort_values(ascending=False).head(top)
            sym = conf.assign(pair=[" ↔ ".join(sorted(p)) for p in zip(conf[own], conf[other])])
            return name, own, other, pairs, sym.pair.value_counts().head(8), len(conf)
    return None


def label_suspects(units, min_score=0.9):
    for name, df in units.items():
        if "outcome" in df and "score" in df and df.outcome.isin(["background", "bkg"]).any():
            bg = df[df.outcome.isin(["background", "bkg"])]
            hi = bg[bg.score >= min_score]
            own = next((c for c in ("category", "label") if c in df), None)
            return name, len(bg), hi, (hi[own].value_counts().head(10) if own else None), min_score
    return None


def iou_sensitivity(units):
    for name, df in units.items():
        if {"iou", "outcome"} <= set(df) and (df.outcome == "mislocalized").any() and "pred_label" in df:
            m = df[df.outcome == "mislocalized"]
            hist = pd.cut(m.iou, [0, 0.1, 0.2, 0.3, 0.4, 0.5]).value_counts().sort_index()
            same = m.pred_label == m.category if "category" in m else pd.Series(True, index=m.index)
            gains = {th: ((m.iou >= th) & same).sum() / len(df) for th in (0.4, 0.3)}
            return name, hist, gains, df.correct.mean(), len(m)
    return None


def concentration(units):
    out = []
    for name, df in units.items():
        if "image_id" not in df:
            continue
        f = (~df.correct).groupby(df.image_id).sum().sort_values(ascending=False)
        if f.sum() == 0:
            continue
        top10 = f.head(max(1, len(f) // 10)).sum() / f.sum()
        per = df.groupby("image_id").correct.agg(["size", "mean"])
        out.append((name, top10, int((per["mean"] == 0).sum()), len(per), f.head(5)))
    return out


# ============================================================
# write
# ============================================================
def summary_lines(results):
    """A few plain sentences: how good is it overall, and how does it usually fail."""
    L = []
    for r in results:
        what, failword = r["words"]
        mix = outcome_mix(r["df"])
        s = f"- **{r['span']}**: {pct(1 - r['base'])} of {len(r['df']):,} {what}s are correct"
        if len(mix):
            s += f"; the commonest failure is *{mix.index[0]}* ({mix.iloc[0]:.0%} of failures)"
            if len(mix) > 1:
                s += f", then *{mix.index[1]}* ({mix.iloc[1]:.0%})"
        if r["cards"]:
            c = max(r["cards"], key=lambda c: c["total_share"])
            s += (f". Biggest problem area: `{c['attribute']} = {c['value']}` ({pct(c['rate'])} {failword} "
                  f"vs {pct(c['rest'])})")
        L.append(s + ".")
    return L + [""]


def card_md(i, c, res):
    what, failword = res["words"]
    span = res["span"]
    L = [f"### {i}. {res['span']}: {c['attribute']} = {c['value']}"]
    L.append(f"**{pct(c['rate'])} {failword}** vs {pct(c['rest'])} for the rest ({c['lift']:.2f}×, "
             f"95% CI {pct(c['ci_low'])}–{pct(c['ci_high'])}). {c['n']:,} {what}s, {c['n_fail']:,} failures, "
             f"**{c['excess']:,.0f} more than expected**.")
    if len(c["others"]):
        o = c["others"].head(8)
        more = f" (+{len(c['others']) - 8} more)" if len(c["others"]) > 8 else ""
        L.append(f"- Other significantly worse values of `{c['attribute']}`: " +
                 ", ".join(f"{v} {pct(rt)} (n={n:,})" for v, rt, n in zip(o.value, o.rate, o.n)) + more +
                 f". All together: {c['total_excess']:,.0f} excess failures = {c['total_share']:.0%} of all {what} failures.")
    else:
        L.append(f"- Share of all {what} failures that this excess represents: {c['total_share']:.0%}.")
    if c.get("also"):
        L.append(f"- Same group of {what}s as: {', '.join(c['also'][:4])}")
    if c["mode"]:
        over = c["mode_share"] / c["mode_base"] if c["mode_base"] else float("nan")
        mix = ", ".join(f"{k} {v:.0%}" for k, v in c["mix"].head(4).items())
        how = (f"mostly **{c['mode']}** ({c['mode_share']:.0%} of failures here vs {c['mode_base']:.0%} overall, "
               f"{over:.1f}× over-represented)" if over > 1.15 else
               f"about the same mix as overall, so this slice fails *more often*, not *differently*")
        L.append(f"- Failure type: {how}. Mix: {mix}.")
    if c["by"] is None:
        L.append(f"- Confounding: no other attribute could be compared ({c['tested']} tested).")
    elif c["retained"] >= 2 / 3:
        L.append(f"- Confounding: **holds** within every other attribute ({c['tested']} tested; the "
                 f"strongest, {c['by']}, still leaves {c['adj']:+.1%} points of the {c['rate'] - c['rest']:+.1%} gap).")
    else:
        L.append(f"- Confounding: **partly explained by {c['by']}**: within {c['by']} the gap shrinks "
                 f"from {c['rate'] - c['rest']:+.1%} to {c['adj']:+.1%} points ({c['retained']:.0%} remains).")
    distinct = bool(c["mode"]) and c["mode_share"] / max(c["mode_base"], 1e-9) > 1.15
    adv = ADVICE.get(c["mode"]) if distinct else None
    if re.match(r"(score|confidence)", c["attribute"]):
        adv = "Low-confidence region: this is a threshold choice, see *Calibration and threshold* below."
    if adv:
        L.append(f"- What to check: {adv}")
    extra = (f" and annotations['outcome'].label == '{c['mode']}'" if distinct
             else " and annotations['correct'].label == 'incorrect'")
    L.append(f"- Phoenix: `{phoenix_filter(span, c['attribute'], c['value'], c['kind'], res['attrs'], extra)}`")
    return "\n".join(L)


def write(path, d, results, units, top):
    L = [f"# Model failure report", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')} from `{d}` "
         f"({', '.join(f'{n}.csv' for n in units)}).", ""]

    # ---- overview
    L += ["## 1. Overview (what Phoenix shows)", "",
          "Each table is one kind of unit, and each row of it one span in Phoenix. The `correct` "
          "annotation summary in Phoenix is the accuracy column below; filter spans and it updates for that slice.", ""]
    rows = []
    for r in results:
        mix = outcome_mix(r["df"])
        rows.append(dict(unit=f"`{r['span']}` ({r['name']}.csv)", rows=f"{len(r['df']):,}",
                         correct=pct(1 - r["base"]),
                         failures=", ".join(f"{k} {v:.0%}" for k, v in mix.items()) or "–"))
    L += [md_table(pd.DataFrame(rows)), ""]

    # ---- headline
    L += summary_lines(results)
    allc = sorted([(c, r) for r in results for c in r["cards"]], key=lambda x: -x[0]["total_share"])
    L += ["## 2. Main problems", ""]
    if not allc:
        L += ["No slice is significantly worse than the rest after correction for multiple testing.", ""]
    else:
        L += ["One entry per attribute, ranked by its excess failures (failures beyond what the overall rate "
              "predicts, summed over all its significantly worse values) as a share of that unit's failures. "
              "Slices that another attribute fully explains are in section 3 instead.", ""]
        L += [f"{i}. **{r['span']}: {c['attribute']} = {c['value']}**"
              f"{' (and ' + str(len(c['others'])) + ' more values)' if len(c['others']) else ''}: "
              f"{pct(c['rate'])} vs {pct(c['rest'])}, {c['total_share']:.0%} of {r['words'][0]} failures are excess"
              f"{', mostly ' + c['mode'] if c['mode'] and c['mode_share'] / max(c['mode_base'], 1e-9) > 1.15 else ''}"
              f"{'' if c['retained'] >= 2 / 3 or c['by'] is None else ' (partly ' + c['by'] + ')'}"
              for i, (c, r) in enumerate(allc[:top], 1)]
        L += [""] + [card_md(i, c, r) + "\n" for i, (c, r) in enumerate(allc[:top], 1)]
        if len(allc) > top:
            L += ["Smaller problems (also significant, not detailed): " +
                  "; ".join(f"{r['span']}: {c['attribute']} = {c['value']} ({pct(c['rate'])} vs {pct(c['rest'])}, "
                            f"{c['total_share']:.0%})" for c, r in allc[top:]) + ".", ""]

    # ---- explained away
    ex = [(c, r) for r in results for c in r["explained"]]
    L += ["## 3. Looks like a problem, but is explained by something else", ""]
    if ex:
        L += ["These slices fail more often overall, but the gap mostly disappears once you compare like with like.", ""]
        L += [f"- **{r['span']}: {c['attribute']} = {c['value']}** ({pct(c['rate'])} vs {pct(c['rest'])}): within "
              f"*{c['by']}* the gap falls from {c['rate'] - c['rest']:+.1%} to {c['adj']:+.1%} points."
              for c, r in sorted(ex, key=lambda x: -x[0]["excess"])[:12]]
    else:
        L += ["None of the top slices is explained away by another attribute."]
    L.append("")

    # ---- calibration
    cal = calibration(units)
    if cal:
        name, sc, tab, sweep, has_recall, smin = cal
        L += ["## 4. Calibration and threshold", "", f"`{name}`: share correct by `{sc}`.", "", md_table(tab), ""]
        if len(sweep):
            s = sweep.copy()
            best = s.loc[s.f1.idxmax()] if has_recall else None
            for c in ("precision", "recall", "f1"):
                if c in s:
                    s[c] = s[c].map(pct)
            L += ["Threshold trade-off" + (" (recall = share of ground truth still found):" if has_recall else ":"), "",
                  md_table(s), ""]
            if best is not None:
                L += [f"Best F1 at score ≥ **{best.threshold:.2f}** (precision {pct(best.precision)}, recall "
                      f"{pct(best.recall)}). Whether that is the right point depends on whether misses or false "
                      "alarms cost more.", ""]
        if smin >= 0.2:
            L += [f"> Note: the lowest score is {smin:.2f}, so predictions were already cut at that threshold. "
                  "Objects \"missed\" may have had a lower-scored box the file does not contain.", ""]

    # ---- confusions
    cf = confusions(units)
    if cf:
        name, own, other, pairs, sym, n = cf
        L += ["## 5. Class confusions", "",
              f"{n:,} {name} rows are in the right place with the wrong class. Most common pairs (`{own}` → `{other}`):", "",
              md_table(pairs.rename("count").reset_index()), "",
              "Both directions combined: " + ", ".join(f"{p} ({v})" for p, v in sym.items()) + ".", "",
              "Pairs that go both ways are often ambiguous in the labels too; check them with the label pipeline.", ""]

    # ---- label suspects
    ls = label_suspects(units)
    if ls:
        name, nbg, hi, by, th = ls
        L += ["## 6. Possible label issues", "",
              f"{len(hi):,} of {nbg:,} detections on unlabelled regions score ≥ {th}. Confident detections with "
              "nothing labelled underneath are often objects missing from the labels, not model errors.", ""]
        if by is not None and len(by):
            L += [md_table(by.rename("count").reset_index()), "",
                  f"Phoenix: `name == '{SPAN_NAMES.get(name, name)}' and annotations['outcome'].label == 'background' "
                  f"and metadata['score'] >= {th}`", ""]

    # ---- iou
    io = iou_sensitivity(units)
    if io:
        name, hist, gains, acc, n = io
        L += ["## 7. Localization sensitivity", "",
              f"{n:,} {name} rows have a box nearby that overlaps too little. Their best IoU:", "",
              md_table(hist.rename("count").rename_axis("IoU").reset_index().astype(str)), "",
              f"Loosening the match would raise the correct rate from {pct(acc)} to "
              f"{pct(acc + gains[0.4])} at IoU 0.4, or {pct(acc + gains[0.3])} at IoU 0.3. "
              + ("Small gain: when boxes are off, they are well off." if gains[0.4] < 0.03 else
                 "Sizeable gain: many boxes are just off; box regression or label box conventions are worth a look."), ""]

    # ---- concentration
    cc = concentration(units)
    if cc:
        L += ["## 8. Where failures concentrate", ""]
        for name, top10, allbad, nimg, worst in cc:
            L.append(f"- `{name}`: the worst 10% of images hold **{top10:.0%}** of failures; {allbad:,} of {nimg:,} "
                     f"images have none of their {UNIT_WORDS.get(name, ('row',))[0]}s correct. Worst images: "
                     + ", ".join(f"{i} ({int(v)})" for i, v in worst.items()) + ".")
        L.append("")

    # ---- per attribute
    L += ["## 9. Every attribute: its worst value", "",
          "The single worst value of each attribute, significant or not (p_adj = Benjamini-Hochberg corrected).", ""]
    for r in results:
        t = r["table"]
        if not len(t):
            continue
        w = t.sort_values("lift", ascending=False).drop_duplicates("attribute").sort_values("excess", ascending=False)
        w = pd.DataFrame(dict(attribute=w.attribute, worst=w.value, n=w.n.map("{:,}".format),
                              fail=w.rate.map(pct), rest=w.rest.map(pct), lift=w.lift.map("{:.2f}".format),
                              excess=w.excess.map("{:,.0f}".format),
                              significant=np.where((w.p_adj < 0.05) & (w.lift > 1), "yes", "no")))
        L += [f"**`{r['span']}`** (overall {pct(r['base'])} {r['words'][1]})", "", md_table(w), ""]

    # ---- caveats
    L += ["## 10. Caveats", "",
          "- Everything here is correlation over this evaluation set. A slice tells you where to look, not why.",
          "- Confounding is checked one attribute at a time; two attributes together can still explain a slice.",
          f"- Slices smaller than the minimum size are not reported; small slices have wide confidence intervals."]
    if any(c.startswith("clip_") for df in units.values() for c in df.columns):
        L.append("- `clip_*` attributes are zero-shot CLIP guesses per image and are noisy (especially focus/lighting).")
    L.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dir", default=os.path.join(HERE, "phoenix_slices"))
    ap.add_argument("--out", default=None, help="default: <dir>/failure_report.md")
    ap.add_argument("--top", type=int, default=8, help="problem cards in section 2")
    ap.add_argument("--min-slice", type=int, default=30)
    ap.add_argument("--skip", default="", help="comma-separated columns not to use as attributes")
    a = ap.parse_args()
    units = load_units(a.dir)
    if not units:
        raise SystemExit(f"No CSV with a `correct` column in {a.dir}")
    skip = {c for c in a.skip.split(",") if c}
    t0 = time.time()
    results = [analyse(n, df, a.min_slice, a.top, skip) for n, df in units.items()]
    out = a.out or os.path.join(a.dir, "failure_report.md")
    write(out, a.dir, results, units, a.top)
    print(f"Wrote {out} ({time.time() - t0:.0f}s): " +
          ", ".join(f"{r['name']} {len(r['cards'])} problems / {len(r['explained'])} explained away" for r in results))


if __name__ == "__main__":
    main()
