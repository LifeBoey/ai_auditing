If you want to plant errors, run in DATA:

python3 _make_perturbed_subset.py --gt <clean gt json> --images <clean image dir> -n <N> \
    --n-missing <n_missing> --n-wrong <n_wrong> \
    --keep-originals --out-dir <perturbed image dir> \
    --out-gt <perturbed ground truth json> --out-log <perturbation log json>

final directory has N+(n_missing)+(n_wrong) images

If you haven't made predictions yet:

python predict.py --images <image dir> --labels <ground truth json> \
    --model <model type> <model path> \
    [repeat --model... as many times as needed]
    --out-dir <prediction json dir> --prefix <your prefix>

To run the actual commands:

python -u run_audit.py --labels <ground truth json> --images-dir <image dir> \
    --predictions <prediction json1>,<prediction json2>,... \
    --budget <number of images in review list>
    --root audit_<your prefix> 2>&1 | tee audit_<your prefix>.log

python coco_review_slideshow.py --csv audit_<your prefix>/ensemble/review_list.csv \
    --labels <ground truth json> \
    --predictions <one of your prediction jsons>
    --score-threshold <score threshold; only bounding boxes with this score and above are shown> --output review_<your prefix>.html

The review html is where you can review the potentially erroneously annotated images. Each image will be complemented with the reasoning from every method used. 

If you ran planted errors:

python analyze_audit_artifacts.py --audit-dir audit_coco500 --perturbation-log <perturbation log json>

The analysis/report.html in the audit-dir is where you can see stats about how each method did in picking up the planted errors. 


====

Worked example:

python3 _make_perturbed_subset.py --gt labels_coco5000.json --images ../DATA/data_sub500_v2 -n 500 -n-missing 25 -n-missing 25 \
    --keep-originals --out-dir data_coco500_perturbed

python predict.py --images ../DATA/data_sub500_v2 --labels ../DATA/labels_sub500_v2.json \
    --model ultralytics /mnt/c/Users/Gabriel/fiftyone/__models__/yolo11x.pt \
    --model frcnn_mbv3  /mnt/c/Users/Gabriel/fiftyone/__models__/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth \
    --out-dir ../DATA/prediction_jsons --prefix coco500
python -u run_audit.py --labels ../DATA/labels_sub500_v2.json --images-dir ../DATA/data_sub500_v2 \
    --predictions ../DATA/prediction_jsons/predictions_coco500_yolo11x.json,../DATA/prediction_jsons/predictions_coco500_frcnn_mbv3.json \
    --root audit_coco500 2>&1 | tee audit_coco500.log

python coco_review_slideshow.py --csv audit_coco500/ensemble/all_ranked.csv \
    --labels ../DATA/labels_sub500_v2.json \
    --predictions ../DATA/prediction_jsons/predictions_coco500_frcnn_mbv3.json \
    --score-threshold 0.5 --output review_coco500_all.html