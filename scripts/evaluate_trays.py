"""
Tray-level pass/fail evaluation with the threshold chosen on validation.

1. Run the detector once on the val and test images, storing each tray's
   highest confidence (results/<run>/max_conf_{val,test}.npy).
2. Choose the highest threshold that reaches --min-recall on VALIDATION.
3. Report test metrics at that fixed threshold with 95 % Wilson intervals,
   plus a full sweep (for the PR trade-off plot) and the missed trays by class.

Example:
    python scripts/evaluate_trays.py --weights runs/yolo11n_grouped/weights/best.pt \
        --data /content/pork_grouped --out results/yolo11n_grouped
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from porkqc import CLASS_NAMES  # noqa: E402
from porkqc.detect import load_model, max_confidences  # noqa: E402
from porkqc.labels import label_path_for, list_images, read_yolo_label  # noqa: E402
from porkqc.tray_eval import (  # noqa: E402
    missed_by_class,
    rates,
    select_threshold,
    sweep,
    tray_confusion,
    wilson_interval,
)


def load_split(data_dir: Path, split: str):
    imgs = list_images(data_dir / split / "images")
    labels = [read_yolo_label(label_path_for(p), len(CLASS_NAMES)) for p in imgs]
    gt = np.array([len(l) > 0 for l in labels])
    classes = [set(l[:, 0].astype(int).tolist()) for l in labels]
    return imgs, gt, classes


def scores_for(model, imgs, cache: Path) -> np.ndarray:
    if cache.exists():
        s = np.load(cache)
        if len(s) == len(imgs):
            return s
    s = max_confidences(model, imgs)
    np.save(cache, s)
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True, help="Dataset folder with val/ and test/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-recall", type=float, default=0.95)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(args.data)
    model = load_model(args.weights)

    v_imgs, v_gt, _ = load_split(data, "val")
    t_imgs, t_gt, t_cls = load_split(data, "test")
    v_s = scores_for(model, v_imgs, out / "max_conf_val.npy")
    t_s = scores_for(model, t_imgs, out / "max_conf_test.npy")

    thr = select_threshold(v_s, v_gt, args.min_recall)
    c = tray_confusion(t_s, t_gt, thr)
    r = rates(c)
    summary = {
        "threshold_from_val": thr,
        "min_recall_target": args.min_recall,
        "test_trays": int(len(t_gt)), "test_defective": int(t_gt.sum()), "test_good": int((~t_gt).sum()),
        **c, **r,
        "recall_95ci": wilson_interval(c["tp"], c["tp"] + c["fn"]),
        "false_alarm_95ci": wilson_interval(c["fp"], c["fp"] + c["tn"]),
        "missed_by_class": {CLASS_NAMES[k]: v for k, v in missed_by_class(t_cls, t_s, thr).items()},
        "missed_files": [p.name for p, g, s in zip(t_imgs, t_gt, t_s) if g and s < thr],
        "false_alarm_files": [p.name for p, g, s in zip(t_imgs, t_gt, t_s) if not g and s >= thr],
    }
    (out / "tray_summary.json").write_text(json.dumps(summary, indent=2))

    rows = sweep(t_s, t_gt, np.round(np.arange(0.05, 0.96, 0.05), 2))
    with open(out / "tray_sweep_test.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
