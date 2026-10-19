"""
Ensemble of annotation-audit black boxes -> one list of images to check by hand.

Every method is a black box that wrote scores.csv + meta.json (see blackbox_common.py);
nothing here knows how a method works. For each image:

  1. each signal (s_*) becomes its percentile rank across images, 0 = least suspicious,
     1 = most; blank stays "no opinion", so scales (probabilities, counts) don't matter
  2. method score = weighted mean of its signals' percentiles
  3. group score  = weighted mean of the methods in the same group (methods built on the same
     model's predictions are correlated, so a group gets one voice); --no-groups: every method votes
  4. the groups' scores are combined with --combine (each group / method with its weight):
        rrf   reciprocal rank fusion, sum w / (k + rank), scaled to 0-1 (default: rewards
              being near the top of several lists, robust to scores that saturate)
        mean  weighted mean group score
        max   best group score (anything any group finds alarming goes to the top)
        vote  weighted fraction of groups whose methods flagged the image (ties broken by mean)

Weights (--weights, comma-separated key=w; everything not mentioned weighs 1, 0 drops it):
  cleanlab=0.5               a method (meta "method") or black box (meta "name", e.g.
                             cleanlab_yolo), within its group; with --no-groups, its vote
  rcnn=2                     a group (grouped mode only)
  fiftyone.s_possible_spurious=0.25   a signal of a method / black box
Signal weights default to the black box's meta "weights"; a name beats a method key.

Writes to --out-dir:
  review_list.csv  the top --budget images: rank, file_path, final score, every rule's score,
                   which methods flagged it, each method's score and reason
  all_ranked.csv   the same for every image
  method_correlation.csv   Spearman correlation between the methods' scores
  evaluation.csv   only with --evaluate-planted (test sets with planted _wrong / _missing images):
                   per method / rule and per kind (wrong, missing, any), how many land in the top
                   100 / budget, ROC AUC, and leave-one-out

Usage:
    python ensemble_audit.py                                  # every scores/*/ with a meta.json
    python ensemble_audit.py scores/flags_rcnn scores/cleanlab_rcnn --combine mean --budget 300
    python ensemble_audit.py --weights "cleanlab=0.25,fiftyone.s_possible_missing=0.5"
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

from blackbox_common import SCORES_ROOT, image_paths, read_blackbox

RULES = ["rrf", "mean", "max", "vote"]
PLANTED = {"wrong": "_wrong", "missing": "_missing"}  # kind: file-name marker of planted test errors


def parse_weights(text):
    weights = {}
    for item in filter(None, (t.strip() for t in (text or "").split(","))):
        key, _, w = item.partition("=")
        try:
            weights[key.strip()] = float(w)
        except ValueError:
            raise SystemExit(f"--weights: '{item}' is not key=number")
    return weights


def lookup(weights, meta, signal=None, default=1.0):
    """Weight for a black box (signal=None) or one of its signals: name beats method."""
    suffix = f".{signal}" if signal else ""
    for key in (meta["name"] + suffix, meta["method"] + suffix):
        if key in weights:
            return weights[key]
    return default


def weighted_mean(df, w):
    """Row-wise mean of df's columns weighted by the Series w, skipping NaN (no opinion)."""
    present = df.notna()
    total = present.mul(w, axis=1).sum(axis=1)
    return (df.fillna(0).mul(w, axis=1).sum(axis=1) / total).where(total > 0)


def load_methods(dirs, names, weights):
    """{method name: (method score Series, flagged Series, reason Series, meta, weight)} over `names`.
    Black boxes with weight 0 are left out."""
    methods = {}
    for d in dirs:
        df, meta = read_blackbox(d)
        df = df.reindex(names)
        if meta["name"] in methods:
            raise SystemExit(f"Two black boxes are both called '{meta['name']}' ({d})")
        weight = lookup(weights, meta)
        signals = [c for c in df.columns if c.startswith("s_")]
        sw = pd.Series({s: lookup(weights, meta, s, meta.get("weights", {}).get(s, 1.0)) for s in signals})
        if weight <= 0 or sw.sum() <= 0:
            print(f"  {meta['name']:<22} weight 0, left out")
            continue
        pct = pd.DataFrame({s: pd.to_numeric(df[s], errors="coerce").rank(pct=True) for s in signals})
        methods[meta["name"]] = (weighted_mean(pct, sw), df["flagged"].fillna(False).astype(bool),
                                 df["reason"].fillna(""), meta, weight)
        sig = ", ".join(f"{s}*{w:g}" for s, w in sw.items())
        print(f"  {meta['name']:<22} w {weight:<4g} group {meta['group']:<8} {sig:<62} "
              f"{int(df['flagged'].sum()):>5} flagged, {int(pct.isna().all(axis=1).sum())} no opinion")
    return methods


def combine(methods, use_groups, k, weights=None):
    """DataFrame with one column per rule, plus the per-unit (group or method) scores."""
    units = {}
    for name, (score, flagged, _, meta, w) in methods.items():
        if use_groups and (weights or {}).get(meta["group"], 1.0) <= 0:
            continue  # group weighted 0: left out, like a method weighted 0
        units.setdefault(meta["group"] if use_groups else name, []).append((score, flagged, w))
    if not units:
        empty = pd.Series(0.0, index=next(iter(methods.values()))[0].index)
        return pd.DataFrame({r: empty for r in RULES}), pd.DataFrame(index=empty.index)
    scores = pd.DataFrame({u: weighted_mean(pd.concat([s for s, _, _ in m], axis=1, ignore_index=True),
                                            pd.Series([w for _, _, w in m])) for u, m in units.items()})
    flags = pd.DataFrame({u: pd.concat([f for _, f, _ in m], axis=1).any(axis=1) for u, m in units.items()})
    # unit weight: the group's own --weights entry, or (--no-groups) the method's weight
    uw = pd.Series({u: (weights or {}).get(u, 1.0) if use_groups else m[0][2] for u, m in units.items()})

    ranks = scores.rank(ascending=False)  # 1 = most suspicious; NaN contributes nothing
    rrf = (1 / (k + ranks)).mul(uw, axis=1).fillna(0).sum(axis=1) / (uw.sum() / (k + 1))
    mean = weighted_mean(scores, uw).fillna(0)
    rules = pd.DataFrame({"rrf": rrf, "mean": mean, "max": scores.loc[:, uw > 0].max(axis=1).fillna(0),
                          "vote": flags.astype(float).mul(uw, axis=1).sum(axis=1) / uw.sum() + 1e-6 * mean})
    return rules, scores


def evaluate(score, planted, budget):
    """{<kind>_top<n>, <kind>_auc} for each planted kind (and "any" if there's more than one)."""
    from sklearn.metrics import roc_auc_score

    order = score.fillna(-1).sort_values(ascending=False, kind="stable").index
    out = {}
    for kind, positives in planted.items():
        negatives = ~planted.drop(columns=kind).any(axis=1) | positives  # other planted kinds don't count
        s, p = score[negatives], positives[negatives]
        for n in sorted({100, budget}):
            out[f"{kind}_top{n}"] = int(positives.loc[order[:n]].sum())
        out[f"{kind}_auc"] = round(roc_auc_score(p, s.fillna(s.min())), 4)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="*", help=f"black-box folders (default: every {SCORES_ROOT}/*/ with a meta.json)")
    parser.add_argument("--labels", default="labels.json", help="the image list")
    parser.add_argument("--images-dir", default="data")
    parser.add_argument("--combine", choices=RULES, default="rrf", help="rule for the final ranking")
    parser.add_argument("--rrf-k", type=float, default=60)
    parser.add_argument("--no-groups", action="store_true", help="every method votes, even if they share a model")
    parser.add_argument("--budget", type=int, default=500, help="images in review_list.csv")
    parser.add_argument("--weights", default="cleanlab=0.5",
                        help="comma-separated key=w for methods, black boxes, groups or <method>.<signal> "
                             "(see above; default: %(default)s)")
    parser.add_argument("--evaluate-planted", action="store_true",
                        help="test sets only: treat *_wrong / *_missing images as planted errors and write "
                             "evaluation.csv (default: off, file names mean nothing)")
    parser.add_argument("--out-dir", default="ensemble_results")
    args = parser.parse_args()
    weights = parse_weights(args.weights)

    dirs = args.dirs or sorted(os.path.dirname(p) for p in glob.glob(os.path.join(SCORES_ROOT, "*", "meta.json")))
    if not dirs:
        raise SystemExit(f"No black boxes found in {SCORES_ROOT}/ (run blackbox_*.py first)")
    paths = image_paths(args.labels, args.images_dir)
    names = list(paths)
    print(f"{len(names)} images, {len(dirs)} black boxes:")
    methods = load_methods(dirs, names, weights)
    rules, units = combine(methods, not args.no_groups, args.rrf_k, weights)
    print(f"Combining {len(units.columns)} {'methods' if args.no_groups else 'groups'} "
          f"({', '.join(units.columns)}) with {args.combine}")

    out = pd.DataFrame({"file_name": names, "file_path": [paths[n] for n in names]}, index=names)
    out["final"] = rules[args.combine]
    for r in RULES:
        out[r] = rules[r].round(4)
    flagged = pd.DataFrame({n: m[1] for n, m in methods.items()})
    out["n_flagged"] = flagged.sum(axis=1)
    out["flagged_by"] = flagged.apply(lambda row: ";".join(row.index[row]), axis=1)
    for n, (score, _, reason, _, _) in methods.items():
        out[n] = score.round(4)
        out[f"{n}_reason"] = reason
    planted = pd.DataFrame(index=names)
    if args.evaluate_planted:  # test sets only: file names mark the planted errors
        planted = pd.DataFrame({kind: [marker in n for n in names] for kind, marker in PLANTED.items()}, index=names)
        planted = planted.loc[:, planted.any()]
        if planted.shape[1] > 1:
            planted["any"] = planted.any(axis=1)
        for kind in planted.columns:
            out[f"planted_{kind}"] = planted[kind]

    out = out.sort_values(["final", "mean"], ascending=False, kind="stable")
    out.insert(0, "rank", range(1, len(out) + 1))
    out["final"] = out["final"].round(4)
    os.makedirs(args.out_dir, exist_ok=True)
    out.to_csv(os.path.join(args.out_dir, "all_ranked.csv"), index=False)
    out.head(args.budget).to_csv(os.path.join(args.out_dir, "review_list.csv"), index=False)
    print(f"Wrote {args.out_dir}/review_list.csv (top {args.budget}) and all_ranked.csv")

    pd.set_option("display.width", 200, "display.max_colwidth", 90)
    print("\nTop 10:")
    for _, r in out.head(10).iterrows():
        why = next((r[f"{n}_reason"] for n in methods if r[f"{n}_reason"]), "")
        print(f"  {r['rank']:>3}  {r['final']:.3f}  {r['file_name']:<28} [{r['flagged_by'] or '-'}]  {why[:90]}")

    print("\nSpearman correlation between methods (high = they find the same images):")
    corr = pd.DataFrame({n: m[0] for n, m in methods.items()}).corr(method="spearman")
    print(corr.round(2).to_string())
    corr.round(4).to_csv(os.path.join(args.out_dir, "method_correlation.csv"))

    if not args.evaluate_planted:
        return
    if planted.empty:
        print(f"\n--evaluate-planted: no file names contain {', '.join(PLANTED.values())}; nothing to evaluate")
        return
    print("\nPlanted errors (" + ", ".join(f"{int(planted[k].sum())} {k}" for k in planted.columns)
          + ") found by each method and rule; AUC per kind is vs clean images only:")
    evals = [{"what": f"method {n}", **evaluate(m[0], planted, args.budget)} for n, m in methods.items()]
    evals += [{"what": f"rule {r}" + (" (final)" if r == args.combine else ""),
               **evaluate(rules[r], planted, args.budget)} for r in RULES]
    if len(units.columns) > 1:
        for u in units.columns:
            rest = {n: m for n, m in methods.items() if (n if args.no_groups else m[3]["group"]) != u}
            loo, _ = combine(rest, not args.no_groups, args.rrf_k, weights)
            evals.append({"what": f"{args.combine} without {u}", **evaluate(loo[args.combine], planted, args.budget)})
    ev = pd.DataFrame(evals)
    print(ev.to_string(index=False))
    ev.to_csv(os.path.join(args.out_dir, "evaluation.csv"), index=False)


if __name__ == "__main__":
    main()
