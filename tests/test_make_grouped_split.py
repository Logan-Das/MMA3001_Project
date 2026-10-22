"""End-to-end test of scripts/make_grouped_split.py on a small fake export.

The fake export copies the structure found in the real Version 4 data:
consecutive frames of one tray, three colour-shifted copies of each training
image, and one frame number that holds two different pictures (regions of
different parts of the scene) placed in different original splits.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_grouped_split  # noqa: E402


def save(img, path):
    Image.fromarray(np.clip(img, 0, 255).astype("uint8")).save(path)


def fake_export(root: Path):
    rng = np.random.default_rng(0)
    splits = ["train", "valid", "test"]
    for s in splits:
        (root / s / "images").mkdir(parents=True)
        (root / s / "labels").mkdir(parents=True)
    k = 0
    for tray in range(10):
        base = rng.random((36, 48)) * 255
        for j in range(4):
            frame = tray * 100 + j
            s = splits[k % 3]
            k += 1
            label = "" if tray in (0, 5) else f"{tray % 5} 0.5 0.5 0.1 0.1\n"
            for copy in range(3 if s == "train" else 1):
                name = f"frame_with_red_{frame}_jpg.rf.{frame:028x}{copy:04x}"
                save(base * (0.85 + 0.1 * copy) + 5 * copy, root / s / "images" / f"{name}.png")
                (root / s / "labels" / f"{name}.txt").write_text(label)
    # A second, different picture under frame 101 (tray 1), in the test split.
    other = rng.random((36, 48)) * 255
    name = f"frame_with_red_101_jpg.rf.{'f' * 32}"
    save(other, root / "test" / "images" / f"{name}.png")
    (root / "test" / "labels" / f"{name}.txt").write_text("3 0.2 0.2 0.1 0.1\n")


def test_split_keeps_distinct_pictures_and_has_no_shared_groups(tmp_path, capsys):
    src, dst = tmp_path / "src", tmp_path / "dst"
    fake_export(src)
    (dst / "train" / "images").mkdir(parents=True)
    (dst / "train" / "images" / "stale.png").write_bytes(b"old run")
    make_grouped_split.main(["--src", str(src), "--dst", str(dst), "--max-gap", "5"])

    out = capsys.readouterr().out
    # 14 train frames x 3 copies + 26 others + 1 extra picture = 69 files
    # -> 41 distinct pictures from 40 frames.
    assert "Removed 28 augmented duplicate images; 41 distinct pictures from 40 frames" in out
    assert "WARNING" not in out
    assert "10 groups" in out
    assert "ORIGINAL split: 10 of 10 groups span 2+ splits" in out
    assert not (dst / "train" / "images" / "stale.png").exists()

    with open(dst / "split_manifest.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 41
    splits_per_group = defaultdict(set)
    for r in rows:
        splits_per_group[r["group"]].add(r["new_split"])
    assert all(len(s) == 1 for s in splits_per_group.values())
    # Both pictures of frame 101 are kept, and in the same split.
    f101 = [r for r in rows if r["frame"] == "101"]
    assert len(f101) == 2 and len({r["new_split"] for r in f101}) == 1

    cfg = yaml.safe_load((dst / "data.yaml").read_text())
    assert cfg["nc"] == 5 and cfg["test"] == "test/images"
    n_imgs = sum(len(list((dst / s / "images").iterdir())) for s in ("train", "val", "test"))
    assert n_imgs == 41
