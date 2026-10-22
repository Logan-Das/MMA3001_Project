"""
Per-tray score table and histogram for one trained model on the test split.

Reads the highest box confidence per test tray saved by
`scripts/evaluate_trays.py` (``max_conf_test.npy``) and the test labels, then
writes:

* ``<prefix>_tray_scores_test.csv``: one row per tray with its file name,
  ground truth (defective or good), the classes it contains, its highest
  confidence and the PASS/FAIL decision at the chosen threshold;
* ``<prefix>_max_conf_histogram.png``: the distribution of highest confidence
  for defective and good trays, with the threshold marked.

The figure shows how well the model separates the two groups and where the
errors sit: defective trays left of the threshold are misses, good trays right
of it are false alarms.

Example (Colab):
    python scripts/plot_tray_scores.py --data /content/pork_grouped \
        --scores /content/drive/MyDrive/pork_rasher/results/yolo11n_grouped/max_conf_test.npy \
        --threshold 0.09 --out results/grouped_split_v4 --prefix yolo11n
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from porkqc import CLASS_NAMES  # noqa: E402
from porkqc.labels import label_path_for, list_images, read_yolo_label  # noqa: E402
from porkqc.tray_eval import rates, tray_confusion  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="Dataset folder containing test/images and test/labels")
    ap.add_argument("--scores", required=True, help="max_conf_test.npy from evaluate_trays.py")
    ap.add_argument("--threshold", type=float, required=True, help="Operating threshold chosen on validation")
    ap.add_argument("--out", required=True, help="Output folder")
    ap.add_argument("--prefix", default="yolo11n", help="File-name prefix, e.g. the model name")
    args = ap.parse_args(argv)

    imgs = list_images(Path(args.data) / "test" / "images")      # same sorted order as evaluate_trays.py
    scores = np.load(args.scores)
    if len(scores) != len(imgs):
        sys.exit(f"{len(scores)} scores but {len(imgs)} test images: the scores belong to a different split")

    labels = [read_yolo_label(label_path_for(p), len(CLASS_NAMES)) for p in imgs]
    gt = np.array([len(l) > 0 for l in labels])
    t = args.threshold
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Per-tray table
    csv_path = out / f"{args.prefix}_tray_scores_test.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "defective", "classes", "max_conf", "decision", "correct"])
        for p, lab, g, s in zip(imgs, labels, gt, scores):
            classes = ";".join(sorted({CLASS_NAMES[int(c)] for c in lab[:, 0]}))
            fail = s >= t
            w.writerow([p.name, int(g), classes, f"{s:.4f}", "FAIL" if fail else "PASS", int(fail == g)])

    # 2. Histogram (matplotlib imported here so the package itself does not need it)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = tray_confusion(scores, gt, t)
    r = rates(c)
    bins = np.linspace(0, 1, 41)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(scores[gt], bins, alpha=0.7, color="#2b6cb0", label=f"defective trays ({gt.sum()})")
    ax.hist(scores[~gt], bins, alpha=0.7, color="#dd8452", label=f"good trays ({(~gt).sum()})")
    ax.axvline(t, color="k", ls="--", label=f"t = {t} (chosen on validation)")
    ax.set_xlabel("highest box confidence in the tray")
    ax.set_ylabel("trays")
    ax.set_title(f"{args.prefix}, test split: {c['fn']} missed defective trays, "
                 f"{c['fp']} false alarms (recall {r['recall']:.2f}, false-alarm rate {r['false_alarm_rate']:.2f})",
                 fontsize=10)
    ax.legend()
    fig.tight_layout()
    png_path = out / f"{args.prefix}_max_conf_histogram.png"
    fig.savefig(png_path, dpi=120)

    print(f"wrote {csv_path} and {png_path}")
    print(f"missed {c['fn']} of {gt.sum()} defective, {c['fp']} false alarms of {(~gt).sum()} good")


if __name__ == "__main__":
    main()
