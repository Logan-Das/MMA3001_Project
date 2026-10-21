"""
Train one detector with the shared experiment settings.

Example:
    python scripts/train.py --model yolo11n.pt --data /content/pork_grouped/data.yaml \
        --out /content/drive/MyDrive/pork_rasher --name yolo11n_grouped
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from porkqc.detect import train  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="e.g. yolo11n.pt or yolo11s.pt")
    ap.add_argument("--data", required=True, help="data.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args(argv)
    train(args.model, args.data, args.out, args.name, epochs=args.epochs)


if __name__ == "__main__":
    main()
