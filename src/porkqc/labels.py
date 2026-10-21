"""
Reading and validating YOLO-format label files.

A YOLO label file holds one line per bounding box::

    <class_id> <x_centre> <y_centre> <width> <height>

where the four box values are normalised to the image size, so each lies in
[0, 1]. An empty file (or a missing file) means the image contains no
defects, i.e. a *good* tray.

Malformed labels are a real risk with exported datasets, so every reader in
this module validates its input and raises `LabelFormatError` with the file
name and line number rather than silently producing wrong ground truth.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


class LabelFormatError(ValueError):
    """Raised when a label file does not follow the YOLO box format."""


def label_path_for(image_path: str | Path) -> Path:
    """Return the label file that belongs to an image in a YOLO export.

    YOLO exports keep images in ``<split>/images/`` and labels in
    ``<split>/labels/`` with the same stem and a ``.txt`` suffix.

    Args:
        image_path: Path to an image inside an ``images`` folder.

    Returns:
        Path to the matching ``.txt`` file in the sibling ``labels`` folder.
        The file may not exist; a missing file is treated as "no defects".

    Raises:
        ValueError: If the image is not inside a folder called ``images``.
    """
    p = Path(image_path)
    if p.parent.name != "images":
        raise ValueError(f"expected image inside an 'images' folder, got {p}")
    return p.parent.parent / "labels" / (p.stem + ".txt")


def read_yolo_label(path: str | Path, num_classes: int = 5) -> np.ndarray:
    """Read and validate one YOLO label file.

    Args:
        path: Path to the ``.txt`` label file. A missing file is allowed and
            returns an empty array (a good tray).
        num_classes: Number of valid classes; class ids must lie in
            ``[0, num_classes)``.

    Returns:
        Array of shape ``(N, 5)``: column 0 is the integer class id, columns
        1-4 are ``x_centre, y_centre, width, height`` in normalised units.
        ``N`` is 0 for a good tray.

    Raises:
        LabelFormatError: If a line does not have five numbers, the class id is
            not a valid integer class, a coordinate is outside [0, 1], or a box
            has zero or negative width or height.
    """
    path = Path(path)
    if not path.exists():
        return np.zeros((0, 5))

    rows = []
    for line_no, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue  # blank lines are harmless
        parts = line.split()
        where = f"{path.name}:{line_no}"
        if len(parts) != 5:
            # Polygon (segmentation) exports have more than 5 values per line.
            raise LabelFormatError(f"{where}: expected 5 values, got {len(parts)}")
        try:
            values = [float(v) for v in parts]
        except ValueError as exc:
            raise LabelFormatError(f"{where}: non-numeric value") from exc
        cls = values[0]
        if cls != int(cls) or not 0 <= cls < num_classes:
            raise LabelFormatError(f"{where}: invalid class id {parts[0]}")
        box = np.array(values[1:])
        if np.any(box < 0) or np.any(box > 1):
            raise LabelFormatError(f"{where}: box values must be in [0, 1]")
        if box[2] <= 0 or box[3] <= 0:
            raise LabelFormatError(f"{where}: box has zero width or height")
        rows.append(values)

    return np.array(rows, dtype=float).reshape(-1, 5)


def list_images(folder: str | Path) -> list[Path]:
    """Return all image files in a folder, sorted by name.

    Args:
        folder: Directory to search (not recursive).

    Returns:
        Sorted list of paths whose suffix is in `IMAGE_SUFFIXES`.

    Raises:
        FileNotFoundError: If the folder does not exist.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"image folder not found: {folder}")
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def class_counts(label_files: Iterable[str | Path], num_classes: int = 5) -> dict:
    """Count boxes per class and images with no defects.

    Args:
        label_files: Label file paths (missing files count as good trays).
        num_classes: Number of valid classes.

    Returns:
        Dict with keys ``"boxes"`` (Counter of class id to box count),
        ``"images"`` (Counter of class id to the number of images containing
        it), ``"background"`` (number of good-tray images) and ``"total"``
        (number of images).
    """
    boxes: Counter = Counter()
    images: Counter = Counter()
    background = total = 0
    for f in label_files:
        arr = read_yolo_label(f, num_classes)
        total += 1
        if len(arr) == 0:
            background += 1
            continue
        ids = arr[:, 0].astype(int)
        boxes.update(ids.tolist())
        images.update(set(ids.tolist()))
    return {"boxes": boxes, "images": images, "background": background, "total": total}
