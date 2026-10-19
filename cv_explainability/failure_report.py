"""
failure_report.py -- write the findings of failure_analysis.py as one Markdown file.

Reads findings.json, overview.json, clusters.json and manifest.json from an artifact folder
and writes <dir>/report.md (everything) and <dir>/report_summary.md (the main problems and
why, plus how to read it). No FiftyOne needed: failure_explorer.py calls both with links into
its saved views, and you can also run this on its own (no links).

Run:  python failure_report.py --dir <artifact dir>
"""

import argparse
import json
import os
import time

VERDICT = {"supports": "✅ supports", "partial": "🟡 partial", "against": "❌ against", "n/a": "· n/a"}
OUTCOME_WORDS = {"detected": "detected", "confused": "found, wrong class", "mislocalized": "right class, bad box",
                 "missed": "missed"}
WHY_WORDS = {"low_score_correct": "proposed correctly but scored under the threshold",
             "low_score_wrong_class": "proposed under the threshold, and as the wrong class",
             "low_score_loc": "only a weak, badly aligned box",
             "shadowed": "next to a same-class detection (likely NMS)",
             "no_proposal": "never proposed at all"}
FP_WORDS = {"bkg": "background (nothing there)", "loc": "right class, bad box", "both": "wrong class and bad box",
            "cls": "right place, wrong class", "dupe": "duplicate of a detected object"}


def finding_codes(findings):
    """{card id: "R1"/"P1"...} in findings.json order. failure_explorer.py uses the same codes."""
    codes = {}
    for side, prefix in (("recall", "R"), ("precision", "P")):
        for k, c in enumerate(findings.get(side, []), 1):
            codes[c["id"]] = f"{prefix}{k}"
    return codes


def _load(adir, name, default=None):
    p = os.path.join(adir, name)
    if not os.path.isfile(p):
        return default
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _pct(x, d=0):
    return "–" if x is None else f"{x:.{d}%}"


def _md(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def _table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(_md(v) for v in r) + " |" for r in rows]
    return out


def _bar(x, width=20):
    n = int(round(max(0.0, min(1.0, x)) * width))
    return "█" * n + "░" * (width - n)


def _link(text, url):
    return f"[{text}]({url})" if url else text


def _card(c, code, links, clusters):
    L = [f"### {code} · {_md(c['label'])}", ""]
    url = links.get(c["id"])
    if url:
        L += [f"[Open in FiftyOne →]({url})", ""]
    if c["id"].split(":", 1)[1].startswith("scene_cluster="):
        cl = clusters.get(c["id"].split("=", 1)[1])
        if cl:
            L += [f"*Scene group: {cl['label']} ({cl['n_images']} images).*", ""]
    L += [f"**{c['headline']}**", ""]
    lift = f"{c['lift']:.1f}x" if c.get("lift") else "–"
    L += [f"{c['n']} units, {c['failures']} failed: **{_pct(c['rate'])}** vs {_pct(c['rate_rest'])} elsewhere "
          f"({lift}); {_pct(c['share_of_failures'])} of all failures."]
    if c.get("modes"):
        L += ["Failures here: " + ", ".join(f"{k} {v}" for k, v in sorted(c["modes"].items(), key=lambda t: -t[1])) + "."]
    if c.get("reasons"):
        L += ["Why missed: " + ", ".join(f"{WHY_WORDS.get(k, k)} {v}" for k, v in
                                        sorted(c["reasons"].items(), key=lambda t: -t[1]) if k) + "."]
    if c.get("aliases"):
        L += ["Also seen as: " + "; ".join(c["aliases"]) + "."]
    L += [""]
    L += _table(["verdict", "possible cause", "evidence"],
                [(VERDICT.get(h["verdict"], h["verdict"]), h["title"], h["evidence"]) for h in c["hypotheses"]])
    acts = [h for h in c["hypotheses"] if h["verdict"] == "supports" and h.get("action")]
    if acts:
        L += ["", "**What to try:**"] + [f"- *{h['title']}*: {h['action']}" for h in acts]
    return L + [""]


def write_report(adir, links=None):
    links = links or {}
    findings = _load(adir, "findings.json")
    ov = _load(adir, "overview.json")
    if findings is None or ov is None:
        raise SystemExit(f"No findings.json / overview.json in {adir}: run failure_analysis.py first")
    man = _load(adir, "manifest.json", {}) or {}
    clusters = {c["id"]: c for c in (_load(adir, "clusters.json", []) or [])}
    codes = finding_codes(findings)
    cfg, n = ov["config"], ov["counts"]

    L = ["# Model failure report", ""]
    model = man.get("model_field") or man.get("model") or "model"
    bits = [f"Model: **{model}**", f"dataset: {man.get('dataset', '?')}"]
    if man.get("suspect_model"):
        bits.append(f"suspect labels from: {man['suspect_model']}")
    L += [", ".join(bits) + ".",
          f"Operating point: score ≥ {cfg['conf_thresh']}, IoU ≥ {cfg['iou_fg']}. "
          f"{n['images']} images, {n['gt']} objects ({n['crowd_ignored']} crowd regions not scored), "
          f"{n['predictions_at_op']} predictions at the operating point.",
          f"Generated {findings.get('generated', time.strftime('%Y-%m-%d %H:%M'))}.", "",
          f"**Recall {_pct(ov['recall'], 1)} · Precision {_pct(ov['precision'], 1)}**", ""]
    if links.get("app"):
        L += [f"Browse everything in FiftyOne: {links['app']} (pick a view from the dropdown).", ""]

    # ---- summary list of findings
    L += ["## Findings at a glance", ""]
    for side, title in (("recall", "Objects the model fails to find"), ("precision", "Detections that are wrong")):
        L += [f"**{title}**", ""]
        rows = []
        for c in findings[side]:
            sup = [h["title"] for h in c["hypotheses"] if h["verdict"] == "supports"][:2]
            label = c["label"]
            if c["id"].split(":", 1)[1].startswith("scene_cluster=") and c["id"].split("=", 1)[1] in clusters:
                label += f" ({clusters[c['id'].split('=', 1)[1]]['label']})"
            rows.append((_link(codes[c["id"]], links.get(c["id"])), label, _pct(c["rate"]),
                         _pct(c["rate_rest"]), _pct(c["share_of_failures"]), "; ".join(sup) or "inconclusive"))
        L += _table(["", "slice", "fails", "elsewhere", "share of failures", "best supported cause"], rows) \
            if rows else ["No significant slices."]
        L += [""]

    # ---- error budget
    L += ["## Where the errors are", "", "**Every object (ground truth)**", ""]
    gt_n = max(n["gt"], 1)
    L += _table(["outcome", "objects", "share"],
                [(OUTCOME_WORDS.get(k, k), v, _pct(v / gt_n)) for k, v in
                 sorted(ov["gt_outcomes"].items(), key=lambda t: -t[1])])
    if ov.get("miss_reasons"):
        tot = max(sum(ov["miss_reasons"].values()), 1)
        L += ["", "**Why missed objects were missed**", ""]
        L += _table(["reason", "objects", "share of misses"],
                    [(WHY_WORDS.get(k, k), v, _pct(v / tot)) for k, v in
                     sorted(ov["miss_reasons"].items(), key=lambda t: -t[1])])
    if ov.get("fp_types"):
        tot = max(sum(ov["fp_types"].values()), 1)
        L += ["", f"**Wrong detections** ({tot} of {n['predictions_at_op']})", ""]
        L += _table(["type", "detections", "share"],
                    [(FP_WORDS.get(k, k), v, _pct(v / tot)) for k, v in sorted(ov["fp_types"].items(), key=lambda t: -t[1])])
    sus = ov.get("suspect") or {}
    if sus.get("n"):
        L += ["", f"{sus['n']} objects carry a suspect-label flag; {_pct(sus['failure_rate_flagged'])} of them "
                  f"\"fail\". Recall without them: {_pct(sus['recall_without'], 1)}."]
    L += [""]

    # ---- cards
    for side, title in (("recall", "Findings: objects the model fails to find"),
                        ("precision", "Findings: detections that are wrong")):
        L += [f"## {title}", ""]
        for c in findings[side]:
            L += _card(c, codes[c["id"]], links, clusters)

    # ---- threshold sweep
    L += ["## Threshold sweep", "", "How recall and precision trade off if the score threshold moves:", ""]
    L += _table(["threshold", "recall", "precision", "correct", "wrong", "background", "bad box", "wrong class", "duplicates"],
                [(("**%s**" % s["thresh"]) if abs(s["thresh"] - cfg["conf_thresh"]) < 1e-9 else s["thresh"],
                  _pct(s["recall"], 1), _pct(s["precision"], 1), s["tp"], s["fp"], s["bkg"],
                  s["loc"] + s["both"], s["cls"], s["dupe"]) for s in ov["sweep"]])
    L += [""]

    # ---- classes
    L += ["## Per class (15 lowest recall)", ""]
    L += _table(["class", "objects", "recall", "detections", "precision", "main failure"],
                [(r["label"], r["n_gt"], _pct(r["recall"]), r["n_pred"], _pct(r["precision"]),
                  OUTCOME_WORDS.get(r["main_failure"], r["main_failure"] or "–")) for r in ov["classes"][:15]])
    if ov.get("confusions"):
        L += ["", "**Most common class confusions** (object found, wrong class)", ""]
        L += _table(["true class", "predicted as", "objects"], [(c["gt"], c["pred"], c["n"]) for c in ov["confusions"][:10]])
    L += [""]

    # ---- dose-response
    L += ["## How failure changes with each measurement", "",
          "Each table splits objects into equal-size bins of one measurement. A smooth ramp suggests "
          "steadily harder cases; a cliff suggests a hard limit (resolution, anchor size, stride).", ""]
    for side, key in (("recall", "curves_recall"), ("precision", "curves_precision")):
        for col, cv in (ov.get(key) or {}).items():
            if cv["shape"] == "flat":
                continue
            what = "objects not found" if side == "recall" else "detections wrong"
            L += [f"**{col}** ({what}): {cv['shape']}"
                  + (f", {cv['direction']}" if cv["shape"] == "smooth" else "")
                  + (f", break near {cv['cliff_at']:.3g}" if cv["shape"] == "cliff" else ""), "", "```"]
            L += [f"{b['lo']:>10.3g} to {b['hi']:<10.3g} {_bar(b['rate'])} {b['rate']:5.0%}  (n={b['n']})"
                  for b in cv["bins"]]
            L += ["```", ""]
    flat = [c for k in ("curves_recall", "curves_precision") for c, cv in (ov.get(k) or {}).items() if cv["shape"] == "flat"]
    if flat:
        L += ["Flat (no effect): " + ", ".join(sorted(set(flat))) + ".", ""]

    # ---- caveats
    L += ["## Read with care", ""] + [f"- {c}" for c in findings.get("caveats", [])]
    nr = next((c["not_run"] for s in ("recall", "precision") for c in findings[s] if c.get("not_run")), [])
    if nr:
        L += ["", "Not run (needs the model itself, not just its predictions):"] + [f"- {x}" for x in nr]
    L += [""]

    path = os.path.join(adir, "report.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print(f"Wrote {path}")
    return path


# ============================================================
# short summary: the main problems and why, plus how to read it
# ============================================================
HOW_TO_READ = """## How to read this summary

**How the findings were produced.** Nothing here is hand-written; it is all counted:

1. **Every box is graded.** Each real object is *detected*, *confused* (found, wrong class),
   *mislocalized* (right class, badly placed box) or *missed*. A missed object also gets a
   reason: *scored too low* (a correct box exists but scored under the threshold), *never
   proposed* (no box there at all), a weak badly-placed box, or *suppressed by a neighbour*.
   Each prediction is *correct*, *background* (nothing there), *bad box*, *wrong class* or
   *duplicate*.
2. **Every box is described** by measurable attributes: class, size, shape (wide = more than
   twice as wide as tall), position in the frame, overlap with other objects, image brightness,
   blur and so on, scene group (images clustered by visual similarity), and whether an
   independent model (the suspect-label model) disagrees with the label.
3. **Groups that fail unusually often are searched for.** Every value of every attribute is a
   candidate group (a *slice*). A slice is kept only if it fails clearly more than the rest of
   the data, the gap survives a statistical correction for testing hundreds of slices, and the
   gap isn't already explained by an earlier finding.
4. **A fixed checklist of causes is run on each slice.** Each check counts something, for
   example "what share of the misses had a correct box scored under the threshold". It
   returns ✅ supports, 🟡 partial, ❌ against, or n/a (couldn't be tested).

**Codes.** **R** = recall side, real objects the model *fails to find*. **P** = precision side,
detections that are *wrong*. The number is the order the search found them in, strongest
first. The full detail for each is in `report.md`; the links open the matching view in the
FiftyOne App.

**Numbers on each problem.**

| term | meaning |
|---|---|
| fails / elsewhere | failure rate inside the slice vs. everywhere else |
| lift | fails ÷ elsewhere (1.3x = 30% more likely to fail) |
| share of failures | fraction of *all* failures on that side that fall in this slice (slices overlap, so these don't add to 100%) |
| Why | the checklist causes that came out ✅, each with the count behind it |
| Ruled out | causes the checklist tested and found ❌ |
| Try | the usual fix for each supported cause |

**Read with care.** This is correlation found in the data, not proof. A ✅ means the numbers
fit that explanation; the cut-offs are rules of thumb. Proving a cause means re-running the
model on altered images (not done here).
"""

SHORT = {"label_issue": "suspect labels", "confounded": "proxy for another attribute",
         "unannotated": "unannotated objects", "background_fp": "fires on background",
         "threshold_effect": "scored too low", "confusion": "class confusion", "duplicates": "duplicates",
         "localization": "badly placed boxes", "crowd_suppression": "suppressed by neighbours",
         "no_proposal": "never proposed", "data_sparsity": "few similar examples",
         "intrinsic": "steady difficulty"}


PLAIN_ATTR = {"area_px": "size", "overlap_max": "overlap with other objects", "center_dist": "distance from the centre",
              "sparsity": "how few similar images exist", "score": "score"}
PLAIN_VALUE = {
    ("position", "edge"): "{u} near the edge of the frame", ("position", "center"): "{u} in the centre of the frame",
    ("position", "middle"): "{u} between the centre and the edge",
    ("aspect_bucket", "wide"): "Wide {u} (more than twice as wide as tall)",
    ("aspect_bucket", "tall"): "Tall {u} (more than twice as tall as wide)",
    ("overlap_bucket", "heavy"): "Heavily overlapped objects (at least 30% covered by other objects)",
    ("overlap_bucket", "light"): "Lightly overlapped objects", ("overlap_bucket", "none"): "Objects with no overlap",
    ("size_bucket", "small"): "Small {u} (under 32×32 px)", ("size_bucket", "medium"): "Medium {u} (32×32 to 96×96 px)",
    ("size_bucket", "large"): "Large {u} (over 96×96 px)",
}


def plain_name(c, clusters):
    """A finding's slice in plain words, e.g. 'box area: Q1 (2.51 to 671)' -> 'Smallest 25% of objects ...'."""
    u = "objects" if c["side"] == "recall" else "detections"
    attr, val = c["id"].split(":", 1)[1].split("=", 1)
    if (attr, val) in PLAIN_VALUE:
        return PLAIN_VALUE[(attr, val)].format(u=u).capitalize()
    if attr == "label":
        return f"Class '{val}'" if c["side"] == "recall" else f"Detections labelled '{val}'"
    if attr == "scene_cluster":
        return f"Scene group {val}" + (f": {clusters[val]['label']}" if val in clusters else "")
    if attr in ("n_obj_bucket", "n_preds_bucket"):
        return f"Images with {val} {'objects' if attr == 'n_obj_bucket' else 'detections'}"
    if attr in ("suspect", "low_quality", "truncated", "is_duplicate") or attr.startswith("issue_"):
        return {"suspect": "Objects with a suspect label", "low_quality": f"{u.capitalize()} in low-quality images",
                "truncated": f"{u.capitalize()} cut off by the image edge",
                "is_duplicate": f"{u.capitalize()} in near-duplicate images"}.get(
            attr, f"{u.capitalize()} in images flagged '{attr[6:].replace('_', ' ')}'")
    if attr.endswith(("_q", "_t")):
        base = attr[:-2]
        name = PLAIN_ATTR.get(base, base.replace("_", " "))
        q = val.split(" ")[0]
        part = {"Q1": "Lowest 25%", "Q2": "Second-lowest 25%", "Q3": "Second-highest 25%", "Q4": "Highest 25%",
                "bottom": "Lowest 10%", "top": "Highest 10%"}.get(q, val)
        if base == "area_px":
            part = part.replace("Lowest", "Smallest").replace("Highest", "Largest").replace("lowest", "smallest") \
                       .replace("highest", "largest")
            rng = val[val.find("(") + 1:val.rfind(")")].replace("<=", "").replace(">=", "").split(" to ")
            try:
                side = [round(float(x) ** 0.5) for x in rng if x.strip()]
                about = (f"about {side[0]}×{side[0]} to {side[1]}×{side[1]} px" if len(side) == 2
                         else f"{'up to' if q == 'bottom' else 'from'} about {side[0]}×{side[0]} px")
                return f"{part} of {u} by size ({about})"
            except ValueError:
                pass
        return f"{part} of {u} by {name} ({val[val.find('(') + 1:val.rfind(')')]})"
    return c["label"]


def _first_sentence(s):
    i = s.find(". ")
    return s if i < 0 else s[:i + 1]


def _is_secondary(c):
    """A finding whose strongest supported cause is 'it's mostly another attribute'."""
    sup = [h["key"] for h in c["hypotheses"] if h["verdict"] == "supports"]
    return bool(sup) and sup[0] == "confounded"


def _problem(k, c, code, links, clusters):
    lift = f", {c['lift']:.1f}x" if c.get("lift") else ""
    L = [f"### {k}. {_md(plain_name(c, clusters))}", "",
         f"*{_link(code, links.get(c['id']))} · slice `{c['label']}`*", "",
         f"- **What:** {_pct(c['rate'])} fail here vs {_pct(c['rate_rest'])} elsewhere{lift}; "
         f"{_pct(c['share_of_failures'])} of all failures ({c['failures']} of {c['n']})."]
    sup = [h for h in c["hypotheses"] if h["verdict"] == "supports" and h["key"] != "confounded"]
    if sup:
        L += ["- **Why:**"] + [f"  - *{h['title']}*: {_first_sentence(h['evidence'])}" for h in sup]
    else:
        part = [h for h in c["hypotheses"] if h["verdict"] == "partial"][:2]
        L += ["- **Why:** no single cause is clearly supported"
              + (f"; partly: {', '.join(h['title'].lower() for h in part)}." if part else ".")]
    out = [SHORT.get(h["key"], h["key"]) for h in c["hypotheses"] if h["verdict"] == "against"]
    if out:
        L += [f"- **Ruled out:** {', '.join(out)}."]
    acts = [h["action"] for h in sup if h.get("action")]
    if acts:
        L += [f"- **Try:** {' '.join(acts)}"]
    return L + [""]


def write_summary(adir, links=None, n_recall=5, n_precision=3):
    links = links or {}
    findings = _load(adir, "findings.json")
    ov = _load(adir, "overview.json")
    if findings is None or ov is None:
        raise SystemExit(f"No findings.json / overview.json in {adir}: run failure_analysis.py first")
    man = _load(adir, "manifest.json", {}) or {}
    clusters = {c["id"]: c for c in (_load(adir, "clusters.json", []) or [])}
    codes = finding_codes(findings)
    cfg, n = ov["config"], ov["counts"]
    model = man.get("model_field") or man.get("model") or "model"

    L = ["# Model failure summary", "",
         f"Model **{model}** on {man.get('dataset', '?')}: {n['images']} images, {n['gt']} objects. "
         f"A detection counts at score ≥ {cfg['conf_thresh']} and box overlap (IoU) ≥ {cfg['iou_fg']}.", "",
         f"**It finds {_pct(ov['recall'])} of objects (recall); {_pct(ov['precision'])} of its detections "
         f"are correct (precision).**", ""]

    # ---- the overall picture, in sentences
    mr, ft = ov.get("miss_reasons") or {}, ov.get("fp_types") or {}
    gto = ov.get("gt_outcomes") or {}
    n_fail = sum(v for k, v in gto.items() if k != "detected")
    L += ["## The big picture", ""]
    if n_fail:
        parts = [f"{OUTCOME_WORDS.get(k, k)}: {_pct(v / n_fail)}" for k, v in
                 sorted(gto.items(), key=lambda t: -t[1]) if k != "detected"]
        L += [f"- **Objects not found ({n_fail}):** " + "; ".join(parts) + "."]
    if mr:
        tot = sum(mr.values())
        top = sorted(mr.items(), key=lambda t: -t[1])[:3]
        L += ["- **Why the missed ones were missed:** " + "; ".join(f"{WHY_WORDS.get(k, k)}: {_pct(v / tot)}"
                                                                 for k, v in top) + "."]
    if ft:
        tot = sum(ft.values())
        top = sorted(ft.items(), key=lambda t: -t[1])[:3]
        L += [f"- **Wrong detections ({tot}):** " + "; ".join(f"{FP_WORDS.get(k, k)}: {_pct(v / tot)}"
                                                              for k, v in top) + "."]
    sw = {round(s["thresh"], 2): s for s in ov.get("sweep", [])}
    if sw:
        lo, hi = sw.get(0.3), sw.get(0.7)
        if lo and hi:
            L += [f"- **Threshold trade-off:** at score ≥ 0.3 recall {_pct(lo['recall'])} / precision "
                  f"{_pct(lo['precision'])}; at ≥ 0.7 recall {_pct(hi['recall'])} / precision {_pct(hi['precision'])}."]
    sus = ov.get("suspect") or {}
    if sus.get("n"):
        L += [f"- **Label quality:** {sus['n']} objects have a suspect label (an independent model disagrees). "
              f"Excluding them, recall is {_pct(sus['recall_without'])}, so labels are not the main problem."
              if (sus["recall_without"] or 0) - ov["recall"] < 0.02 else
              f"- **Label quality:** {sus['n']} objects have a suspect label; excluding them recall rises to "
              f"{_pct(sus['recall_without'])}. Fix those labels first."]
    L += [""]

    # ---- main problems
    for side, title, k_max in (("recall", "Main problems: objects the model fails to find", n_recall),
                               ("precision", "Main problems: detections that are wrong", n_precision)):
        cards = findings.get(side, [])
        main = [c for c in cards if not _is_secondary(c)][:k_max]
        rest = [c for c in cards if c not in main]
        L += [f"## {title}", ""]
        if not main:
            L += ["No significant problem groups found.", ""]
        for k, c in enumerate(main, 1):
            L += _problem(k, c, codes[c["id"]], links, clusters)
        if rest:
            L += ["**Also flagged** (mostly the same failures seen through another attribute, or smaller):", ""]
            for c in rest:
                sup = [SHORT.get(h["key"], h["key"]) for h in c["hypotheses"] if h["verdict"] == "supports"][:2]
                L += [f"- {_link(codes[c['id']], links.get(c['id']))} {_md(plain_name(c, clusters))}: "
                      f"{_pct(c['rate'])} vs {_pct(c['rate_rest'])}; {', '.join(sup) or 'inconclusive'}."]
            L += [""]

    L += [HOW_TO_READ, "", "Full detail, every check and every table: `report.md`.", ""]
    path = os.path.join(adir, "report_summary.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print(f"Wrote {path}")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dir", required=True)
    d = ap.parse_args().dir
    write_report(d)
    write_summary(d)
