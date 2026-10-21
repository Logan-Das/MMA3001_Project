"""End-to-end test of scripts/make_grouped_split.py on a small fake export."""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_grouped_split  # noqa: E402


def fake_export(root: Path):
    """Ten 'trays' of 4 consecutive frames, scattered randomly across splits
    the way a random split would (i.e. with leakage)."""
    splits = ["train", "valid", "test"]
    for s in splits:
        (root / s / "images").mkdir(parents=True)
        (root / s / "labels").mkdir(parents=True)
    k = 0
    for tray in range(10):
        for j in range(4):
            frame = tray * 100 + j
            s = splits[k % 3]
            k += 1
            label = "" if tray in (0, 5) else f"{tray % 5} 0.5 0.5 0.1 0.1\n"
            # Train frames get 3 augmented copies (same frame, different hash),
            # as in the Roboflow V4 export.
            for copy in range(3 if s == "train" else 1):
                name = f"frame_with_red_{frame}_jpg.rf.{frame:028x}{copy:04x}"
                (root / s / "images" / f"{name}.jpg").write_bytes(b"x")
                (root / s / "labels" / f"{name}.txt").write_text(label)


def test_split_has_no_shared_groups(tmp_path, capsys):
    src, dst = tmp_path / "src", tmp_path / "dst"
    fake_export(src)
    make_grouped_split.main(["--src", str(src), "--dst", str(dst), "--max-gap", "5"])

    out = capsys.readouterr().out
    # 14 train frames x 3 copies + 26 others = 68 files -> 40 unique frames.
    assert "Removed 28 augmented duplicate images; 40 unique frames remain" in out
    assert "WARNING" not in out
    assert "10 groups" in out
    assert "ORIGINAL split: 10 of 10 groups span 2+ splits" in out

    with open(dst / "split_manifest.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 40
    splits_per_group = defaultdict(set)
    for r in rows:
        splits_per_group[r["group"]].add(r["new_split"])
    assert all(len(s) == 1 for s in splits_per_group.values())

    cfg = yaml.safe_load((dst / "data.yaml").read_text())
    assert cfg["nc"] == 5 and cfg["test"] == "test/images"
    n_imgs = sum(len(list((dst / s / "images").iterdir())) for s in ("train", "val", "test"))
    assert n_imgs == 40
