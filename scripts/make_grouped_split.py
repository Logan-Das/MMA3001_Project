"""
Build a leakage-free train/val/test split from a Roboflow YOLO export.

Use an export made WITHOUT augmentation, so each original image appears once.

Steps:
  1. Pool every image from the export's train/valid/test folders.
  2. Parse the frame number from each file name and print the gap percentiles.
  3. Report how many frame groups the ORIGINAL Roboflow split shares between
     splits (the leakage evidence for the report).
  4. Assign whole groups to new splits, balancing class counts.
  5. Copy images and labels into a new dataset folder with a data.yaml and a
     split_manifest.csv recording every decision.

Example (Colab):
    python scripts/make_grouped_split.py \
        --src "/content/Pork-Rasher-Error(Packaging)-5" \
        --dst /content/pork_grouped --max-gap 5 --seed 0

Run with --dry-run first to see the gap percentiles and choose --max-gap.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from porkqc import CLASS_NAMES  # noqa: E402
from porkqc.grouping import (  # noqa: E402
    gap_percentiles,
    group_by_frame_gap,
    grouped_split,
    groups_spanning_splits,
    parse_frame_id,
)
from porkqc.labels import label_path_for, list_images, read_yolo_label  # noqa: E402

SPLIT_DIRS = ("train", "valid", "test")
SPLIT_NAMES = ("train", "val", "test")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="Roboflow YOLO export folder (no augmentation)")
    ap.add_argument("--dst", help="Output dataset folder")
    ap.add_argument("--max-gap", type=int, default=5, help="Largest frame gap treated as the same tray")
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="Only print statistics")
    args = ap.parse_args(argv)

    src = Path(args.src)
    images, orig_split = [], []
    for s_idx, d in enumerate(SPLIT_DIRS):
        folder = src / d / "images"
        if folder.is_dir():
            imgs = list_images(folder)
            images += imgs
            orig_split += [s_idx] * len(imgs)
    if not images:
        sys.exit(f"No images found under {src}/{{train,valid,test}}/images")

    # 1-2. Parse frame ids; refuse to guess for names that do not parse.
    fids, bad = [], []
    for p in images:
        try:
            fids.append(parse_frame_id(p.name))
        except ValueError:
            bad.append(p.name)
    if bad:
        sys.exit(f"{len(bad)} file names have no frame number, e.g. {bad[:3]}. Inspect them first.")

    frames_by_stem: dict[str, list[int]] = {}
    for f in fids:
        frames_by_stem.setdefault(f.stem, []).append(f.frame)
    print(f"{len(images)} images from {len(frames_by_stem)} source(s)")
    for stem, frames in frames_by_stem.items():
        print(f"  {stem}: {len(frames)} frames, range {min(frames)}-{max(frames)}")
        print(f"    gap percentiles: {gap_percentiles(frames)}")

    groups = group_by_frame_gap(fids, args.max_gap)
    n_groups = len(set(groups))
    print(f"\nmax_gap={args.max_gap}: {n_groups} groups (mean {len(images) / n_groups:.1f} images/group)")

    # 3. Leakage in the original split.
    shared = groups_spanning_splits(groups, orig_split)
    n_imgs_shared = sum(g in set(shared) for g in groups)
    print(f"ORIGINAL split: {len(shared)} of {n_groups} groups span 2+ splits "
          f"({n_imgs_shared} images affected)")

    # 4. Profiles: [images, background images, boxes per class].
    k = len(CLASS_NAMES)
    labels = [read_yolo_label(label_path_for(p), k) for p in images]
    profiles: dict[str, np.ndarray] = {}
    for g, lab in zip(groups, labels):
        v = np.zeros(2 + k)
        v[0] = 1
        v[1] = len(lab) == 0
        for c in lab[:, 0].astype(int):
            v[2 + c] += 1
        profiles[g] = profiles.get(g, np.zeros(2 + k)) + v
    assign = grouped_split(profiles, args.ratios, args.seed)
    new_split = [assign[g] for g in groups]
    assert not groups_spanning_splits(groups, new_split), "grouped split leaked"

    header = ["split", "images", "good"] + list(CLASS_NAMES)
    print("\nNEW split (box counts per class):")
    print("  " + "  ".join(f"{h:>15}" for h in header))
    for s in range(3):
        tot = sum(profiles[g] for g, a in assign.items() if a == s)
        print("  " + "  ".join(f"{x:>15}" for x in [SPLIT_NAMES[s]] + [int(v) for v in tot]))

    if args.dry_run or not args.dst:
        return

    # 5. Write the dataset.
    dst = Path(args.dst)
    for s in SPLIT_NAMES:
        (dst / s / "images").mkdir(parents=True, exist_ok=True)
        (dst / s / "labels").mkdir(parents=True, exist_ok=True)
    with open(dst / "split_manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "stem", "frame", "group", "original_split", "new_split", "n_boxes"])
        for p, f, g, o, n, lab in zip(images, fids, groups, orig_split, new_split, labels):
            s = SPLIT_NAMES[n]
            shutil.copy2(p, dst / s / "images" / p.name)
            lp = label_path_for(p)
            if lp.exists():
                shutil.copy2(lp, dst / s / "labels" / lp.name)
            w.writerow([p.name, f.stem, f.frame, g, SPLIT_NAMES[o], s, len(lab)])
    cfg = {"path": str(dst), "train": "train/images", "val": "val/images", "test": "test/images",
           "nc": k, "names": list(CLASS_NAMES)}
    (dst / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"\nWrote {dst} (data.yaml, split_manifest.csv)")


if __name__ == "__main__":
    main()
