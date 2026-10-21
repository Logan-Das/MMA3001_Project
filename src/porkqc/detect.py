"""
Thin wrapper around the Ultralytics YOLO detector.

Ultralytics is imported inside each function, so importing `porkqc` (and
running the unit tests) does not need a GPU or the Ultralytics package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def load_model(weights: str | Path):
    """Load a YOLO model.

    Args:
        weights: A ``.pt`` checkpoint (e.g. ``best.pt``) or a model name such
            as ``"yolo11n.pt"`` to download pretrained COCO weights.

    Returns:
        An ``ultralytics.YOLO`` instance.
    """
    from ultralytics import YOLO

    return YOLO(str(weights))


def train(model_name: str, data_yaml: str | Path, out_dir: str | Path, run_name: str, **overrides):
    """Train a detector with the settings shared by every experiment.

    Keeping the defaults in one place guarantees that model comparisons differ
    only in the model, not in silently different hyperparameters.

    Args:
        model_name: Starting weights, e.g. ``"yolo11n.pt"`` or ``"yolo11s.pt"``.
        data_yaml: Dataset description file.
        out_dir: Folder for results (put this on Google Drive in Colab).
        run_name: Sub-folder name for this run.
        **overrides: Any Ultralytics training argument to change.

    Returns:
        The trained ``ultralytics.YOLO`` model.
    """
    settings = dict(epochs=100, imgsz=640, batch=16, seed=0, deterministic=True, patience=30)
    settings.update(overrides)
    model = load_model(model_name)
    model.train(data=str(data_yaml), project=str(out_dir), name=run_name, **settings)
    return model


def max_confidences(model, image_paths: Sequence[str | Path], min_conf: float = 0.01) -> np.ndarray:
    """Highest detection confidence per image (0 if nothing is detected).

    Run once at a low confidence floor and store the result; every threshold
    in a sweep can then be evaluated without re-running the model.

    Args:
        model: A loaded ``ultralytics.YOLO`` model.
        image_paths: Images to process.
        min_conf: Detections below this are discarded by the detector. Must
            be lower than any threshold you intend to evaluate.

    Returns:
        Array of shape ``(len(image_paths),)``.
    """
    out = []
    for r in model.predict([str(p) for p in image_paths], conf=min_conf, verbose=False, stream=True):
        out.append(float(r.boxes.conf.max()) if len(r.boxes) else 0.0)
    return np.array(out)
