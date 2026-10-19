#!/usr/bin/env python3
import argparse
import base64
import csv
import json
import mimetypes
import shutil
from pathlib import Path
from urllib.parse import quote

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def norm_name(value):
    return Path(str(value).replace("\\\\", "/")).name

def resolve_path(file_path, image_root=None):
    p = Path(file_path).expanduser()
    if p.exists():
        return p.resolve()
    if image_root:
        root = Path(image_root).expanduser()
        for candidate in (root / file_path.lstrip("/\\\\"), root / norm_name(file_path)):
            if candidate.exists():
                return candidate.resolve()
    return p

def index_coco(coco):
    anns_by_image = {}
    for ann in coco.get("annotations", []):
        anns_by_image.setdefault(ann.get("image_id"), []).append(ann)
    cats = {c["id"]: c.get("name", str(c["id"])) for c in coco.get("categories", [])}
    by_filename = {norm_name(img.get("file_name", "")): img for img in coco.get("images", [])}
    return anns_by_image, cats, by_filename

def annotations_for(path, coco, index, score_threshold=None):
    anns_by_image, cats, by_filename = index
    img = by_filename.get(norm_name(path))
    if img is None:
        return [], None
    result = []
    for ann in anns_by_image.get(img["id"], []):
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        score = ann.get("score")
        if score_threshold is not None and score is not None and float(score) < score_threshold:
            continue
        result.append({
            "bbox": bbox,
            "category_id": ann.get("category_id"),
            "category": cats.get(ann.get("category_id"), f"class {ann.get('category_id')}"),
            "score": ann.get("score"),
        })
    return result, img

def image_to_data_url(path):
    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return "data:{};base64,{}".format(mime, base64.b64encode(path.read_bytes()).decode("ascii"))

def main():
    parser = argparse.ArgumentParser(description="Create an interactive COCO prediction review slideshow.")
    parser.add_argument("--csv", default="review_list.csv")
    parser.add_argument("--predictions", default="predictions.json")
    parser.add_argument("--labels", default="labels.json")
    parser.add_argument("--output", default="review.html")
    parser.add_argument("--image-root", default=None, help="Optional root for resolving image paths.")
    parser.add_argument("--copy-images-to", default=None,
                        help="Copy images to this folder and reference them from HTML instead of embedding.")
    parser.add_argument("--score-threshold", type=float, default=None,
                        help="Hide prediction boxes with a score below this value.")
    args = parser.parse_args()

    predictions = load_json(args.predictions)
    labels = load_json(args.labels)
    pred_index = index_coco(predictions)
    gt_index = index_coco(labels)

    with open(args.csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "file_path" not in fieldnames:
            raise ValueError("CSV must contain a column named 'file_path'.")
        techniques = [c for c in fieldnames if f"{c}_reason" in fieldnames]
        rows = [row for row in reader if row.get("file_path", "").strip()]
    paths = [row["file_path"].strip() for row in rows]
    if not paths:
        raise ValueError("No non-empty file_path values found in CSV.")

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    copy_dir = Path(args.copy_images_to).expanduser().resolve() if args.copy_images_to else None
    if copy_dir:
        copy_dir.mkdir(parents=True, exist_ok=True)

    slides, missing = [], []
    for i, (raw_path, row) in enumerate(zip(paths, rows)):
        image_path = resolve_path(raw_path, args.image_root)
        if not image_path.exists():
            missing.append(raw_path)
            continue

        pred_anns, pred_img = annotations_for(raw_path, predictions, pred_index, args.score_threshold)
        gt_anns, gt_img = annotations_for(raw_path, labels, gt_index)

        if copy_dir:
            target = copy_dir / f"{i:06d}_{image_path.name}"
            if image_path.resolve() != target.resolve():
                shutil.copy2(image_path, target)
            try:
                src = quote(str(target.relative_to(output_path.parent)))
            except ValueError:
                src = target.as_uri()
        else:
            src = image_to_data_url(image_path)

        slides.append({
            "path": raw_path, "src": src,
            "predictions": pred_anns, "ground_truth": gt_anns,
            "pred_image_id": pred_img.get("id") if pred_img else None,
            "gt_image_id": gt_img.get("id") if gt_img else None,
            "width": (gt_img or pred_img or {}).get("width"),
            "height": (gt_img or pred_img or {}).get("height"),
            "techniques": [{"technique": t, "score": row.get(t, ""), "reason": row.get(f"{t}_reason", "")}
                           for t in techniques],
        })

    payload = json.dumps(slides, ensure_ascii=False).replace("</", "<\\/")
    html = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>COCO Prediction Review</title>
<style>
:root{color-scheme:dark;--bg:#111318;--panel:#1b1e26;--line:#343946;--text:#eef0f5;--muted:#aeb4c2}
*{box-sizing:border-box}body{margin:0;font:14px/1.4 system-ui,sans-serif;background:var(--bg);color:var(--text)}
header{position:sticky;top:0;z-index:5;display:flex;flex-wrap:wrap;align-items:center;gap:10px;padding:10px 14px;background:#191c23;border-bottom:1px solid var(--line)}
button,input{background:#292e39;color:var(--text);border:1px solid #454c5b;border-radius:6px;padding:7px 10px;font:inherit}
button{cursor:pointer}button:hover{background:#373e4b}input[type=number]{width:80px}input[type=search]{width:min(300px,60vw)}
.check{display:flex;align-items:center;gap:5px;white-space:nowrap}.layout{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:12px;padding:12px}
.viewer{min-width:0;background:#08090c;border:1px solid var(--line);border-radius:8px;min-height:65vh;display:flex;align-items:center;justify-content:center;overflow:auto}
.stage{position:relative;display:inline-block;line-height:0;max-width:100%}
#photo{display:block;max-width:calc(100vw - 360px);max-height:calc(100vh - 150px);width:auto;height:auto;object-fit:contain}
#overlay{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
aside{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px;overflow-wrap:anywhere}
h2{font-size:15px;margin:0 0 8px}h3{font-size:13px;margin:14px 0 6px}.muted,.help{color:var(--muted)}
.legend{display:flex;flex-wrap:wrap;gap:10px;margin:8px 0}.swatch{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:5px}
.ann{border-top:1px solid var(--line);padding:7px 0;font-size:12px}.path{max-width:50vw;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.help{padding:0 14px 12px}
.techs{width:100%;margin-top:12px;border-collapse:collapse;background:var(--panel);border:1px solid var(--line);font-size:13px}
.techs th,.techs td{border:1px solid var(--line);padding:6px 10px;text-align:left;vertical-align:top}
.techs th{background:#232733}.techs td:nth-child(2){white-space:nowrap;font-variant-numeric:tabular-nums}
@media(max-width:900px){.layout{grid-template-columns:1fr}#photo{max-width:calc(100vw - 32px);max-height:65vh}.path{max-width:80vw}}
</style></head>
<body>
<header>
<button id="prev">← Previous</button><button id="next">Next →</button><span id="counter"></span>
<label>Go to <input id="jump" type="number" min="1" value="1"></label>
<input id="search" type="search" placeholder="Filter by file path…">
<label class="check"><input id="showPred" type="checkbox" checked> Predictions</label>
<label class="check"><input id="showGt" type="checkbox" checked> Ground truth</label>
<label class="check"><input id="showLabels" type="checkbox" checked> Box labels</label>
<button id="fit">Fit image</button></header>
<div class="layout"><main><div class="viewer"><div class="stage"><img id="photo" alt="Review image"><canvas id="overlay"></canvas></div></div>
<table class="techs"><thead><tr><th>Technique</th><th>Score</th><th>Reason</th></tr></thead><tbody id="techBody"></tbody></table>
<div class="help">Keyboard: ← / → previous/next, Home / End first/last, + / − zoom, P toggle predictions, G toggle ground truth.</div></main>
<aside><h2 id="filename">Image details</h2><div class="muted" id="dimensions"></div>
<div class="legend"><span><i class="swatch" style="background:#ff4d4d"></i>Prediction</span><span><i class="swatch" style="background:#38d996"></i>Ground truth</span></div>
<h3>Predictions <span id="predCount"></span></h3><div id="predList"></div>
<h3>Ground truth <span id="gtCount"></span></h3><div id="gtList"></div></aside></div>
<script>
const slides = __SLIDES__;
let filtered=slides.map((_,i)=>i), position=0;
const photo=document.getElementById('photo'), canvas=document.getElementById('overlay'), ctx=canvas.getContext('2d');
function current(){return slides[filtered[position]]}
function fmt(n){return Number(n).toFixed(1)}
function resizeCanvas(){
 const r=photo.getBoundingClientRect(); if(!r.width||!r.height)return;
 const dpr=window.devicePixelRatio||1;canvas.width=Math.round(r.width*dpr);canvas.height=Math.round(r.height*dpr);
 canvas.style.width=r.width+'px';canvas.style.height=r.height+'px';ctx.setTransform(dpr,0,0,dpr,0,0);draw();
}
function drawGroup(anns,color,prefix,iw,ih){
 const sx=photo.clientWidth/(iw||photo.naturalWidth),sy=photo.clientHeight/(ih||photo.naturalHeight);
 ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=2;ctx.font='12px system-ui,sans-serif';
 anns.forEach(a=>{if(!a.bbox||a.bbox.length!==4)return;const [x,y,w,h]=a.bbox,xx=x*sx,yy=y*sy,ww=w*sx,hh=h*sy;ctx.strokeRect(xx,yy,ww,hh);
 if(document.getElementById('showLabels').checked){let label=prefix+': '+(a.category||('class '+a.category_id));if(a.score!==null&&a.score!==undefined)label+=' '+Number(a.score).toFixed(2);
 const tw=ctx.measureText(label).width,ty=Math.max(0,yy-18);ctx.fillRect(xx,ty,tw+8,17);ctx.fillStyle='#101216';ctx.fillText(label,xx+4,ty+12);ctx.fillStyle=color;}});
}
function draw(){
 if(!photo.complete||!photo.naturalWidth)return;const s=current();if(!s)return;
 ctx.clearRect(0,0,canvas.width,canvas.height);const iw=s.width||photo.naturalWidth,ih=s.height||photo.naturalHeight;
 if(document.getElementById('showPred').checked)drawGroup(s.predictions,'#ff4d4d','P',iw,ih);
 if(document.getElementById('showGt').checked)drawGroup(s.ground_truth,'#38d996','GT',iw,ih);
}
function renderList(id,anns,prefix){
 const el=document.getElementById(id);el.innerHTML='';
 if(!anns.length){el.textContent='No annotations found.';el.className='muted ann';return}
 el.className='';anns.forEach((a,i)=>{const d=document.createElement('div');d.className='ann';const b=a.bbox||[];
 const score=a.score===null||a.score===undefined?'':' · score '+Number(a.score).toFixed(3);
 d.textContent=prefix+(i+1)+'. '+(a.category||('class '+a.category_id))+score+' · bbox ['+b.map(fmt).join(', ')+']';el.appendChild(d)});
}
function renderTechniques(rows){
 const body=document.getElementById('techBody');body.innerHTML='';
 (rows||[]).forEach(r=>{const tr=document.createElement('tr');[r.technique,r.score,r.reason].forEach(v=>{const td=document.createElement('td');td.textContent=v??'';tr.appendChild(td)});body.appendChild(tr)});
}
function render(){
 if(!filtered.length){renderTechniques([]);photo.removeAttribute('src');document.getElementById('counter').textContent='0 images';document.getElementById('filename').textContent='No matching images';document.getElementById('predList').innerHTML='';document.getElementById('gtList').innerHTML='';ctx.clearRect(0,0,canvas.width,canvas.height);return}
 position=Math.max(0,Math.min(position,filtered.length-1));const s=current();
 photo.onload=()=>{resizeCanvas();document.getElementById('dimensions').textContent=photo.naturalWidth+' × '+photo.naturalHeight+' px · prediction image_id: '+(s.pred_image_id??'not matched')+' · label image_id: '+(s.gt_image_id??'not matched')};
 photo.src=s.src;document.getElementById('counter').textContent=(position+1)+' / '+filtered.length;document.getElementById('jump').value=position+1;
 document.getElementById('filename').textContent=s.path;document.getElementById('predCount').textContent='('+s.predictions.length+')';document.getElementById('gtCount').textContent='('+s.ground_truth.length+')';
 renderList('predList',s.predictions,'P');renderList('gtList',s.ground_truth,'GT');renderTechniques(s.techniques);draw();
}
function move(n){position+=n;render()}
document.getElementById('prev').onclick=()=>move(-1);document.getElementById('next').onclick=()=>move(1);
document.getElementById('jump').onchange=e=>{position=Number(e.target.value||1)-1;render()};
['showPred','showGt','showLabels'].forEach(id=>document.getElementById(id).onchange=draw);
document.getElementById('search').oninput=e=>{const q=e.target.value.toLowerCase(),old=filtered[position];filtered=slides.map((s,i)=>({s,i})).filter(x=>x.s.path.toLowerCase().includes(q)).map(x=>x.i);const found=filtered.indexOf(old);position=found>=0?found:0;render()};
document.getElementById('fit').onclick=()=>{photo.style.width='';photo.style.height='';photo.style.maxWidth=innerWidth<900?'calc(100vw - 32px)':'calc(100vw - 360px)';photo.style.maxHeight=innerWidth<900?'65vh':'calc(100vh - 150px)';resizeCanvas()};
window.addEventListener('resize',resizeCanvas);
window.addEventListener('keydown',e=>{if(['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName))return;
 if(e.key==='ArrowLeft')move(-1);else if(e.key==='ArrowRight')move(1);else if(e.key==='Home'){position=0;render()}else if(e.key==='End'){position=filtered.length-1;render()}
 else if(e.key.toLowerCase()==='p'){const x=document.getElementById('showPred');x.checked=!x.checked;draw()}else if(e.key.toLowerCase()==='g'){const x=document.getElementById('showGt');x.checked=!x.checked;draw()}
 else if(e.key==='+'||e.key==='='){photo.style.maxWidth='none';photo.style.maxHeight='none';photo.style.width=(photo.clientWidth*1.2)+'px';resizeCanvas()}
 else if(e.key==='-'){photo.style.maxWidth='none';photo.style.maxHeight='none';photo.style.width=(photo.clientWidth/1.2)+'px';resizeCanvas()}});
render();
</script></body></html>'''
    html = html.replace("__SLIDES__", payload)
    output_path.write_text(html, encoding="utf-8")
    print(f"Wrote: {output_path}")
    print(f"Images included: {len(slides)} / {len(paths)}")
    if missing:
        print(f"Missing images: {len(missing)}")
        for item in missing[:20]:
            print("  - " + item)
    if not slides:
        raise SystemExit("No images found; check file_path values and --image-root.")

if __name__ == "__main__":
    main()
