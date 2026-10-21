"""
Leakage control: grouping neighbouring video frames and splitting by group.

Why this module exists
----------------------
The dataset images are frames sampled from video of a packing line, with names
such as ``frame_with_red_121_jpg.rf.<hash>.jpg``. Frames that are close in time
show the *same* tray from almost the same viewpoint. Roboflow's default split
assigns images to train/val/test at random, so near-identical frames of one
tray can land in both the training and the test split. The model is then
partly tested on trays it has already seen, and the test score overstates how
it would perform on new trays (data leakage).

The fix implemented here:

1. Parse the source video stem and frame number from each file name
   (`parse_frame_id`).
2. Sort frames and start a new group whenever the gap to the previous frame
   exceeds a threshold (`group_by_frame_gap`). The threshold is chosen from
   the distribution of gaps (`gap_percentiles`).
3. Assign whole groups to splits so no group spans two splits, while keeping
   the class mix of each split close to the target ratios (`grouped_split`).
4. Independently check the result with a perceptual hash, counting test images
   that have a near-identical image in train (`dhash`,
   `cross_split_near_duplicates`). This check does not rely on file names, so
   it also catches duplicates the frame-number rule misses.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

# Roboflow export name: <stem>_<frame>_<ext>.rf.<hash>.<ext>
_RF_PATTERN = re.compile(
    r"^(?P<stem>.+?)_(?P<frame>\d+)_(?:jpe?g|png)\.rf\.[0-9a-fA-F]+\.(?:jpe?g|png)$"
)
# Plain name: <stem>_<frame>.<ext>
_PLAIN_PATTERN = re.compile(r"^(?P<stem>.+?)_(?P<frame>\d+)\.(?:jpe?g|png)$")


@dataclass(frozen=True)
class FrameId:
    """Identity of a video frame recovered from a file name.

    Attributes:
        stem: Name of the source video or capture session, e.g.
            ``"frame_with_red"``.
        frame: Integer frame number within that source.
    """

    stem: str
    frame: int


def parse_frame_id(filename: str | Path) -> FrameId:
    """Recover the source stem and frame number from an image file name.

    Handles both Roboflow export names
    (``frame_with_red_121_jpg.rf.ad75...jpg``) and plain names
    (``frame_with_red_121.jpg``).

    Args:
        filename: File name or path. Only the final component is used.

    Returns:
        The parsed `FrameId`.

    Raises:
        ValueError: If the name does not contain a frame number in either
            format. Such images cannot be grouped safely and should be
            inspected by hand rather than guessed.
    """
    name = Path(filename).name
    m = _RF_PATTERN.match(name) or _PLAIN_PATTERN.match(name)
    if m is None:
        raise ValueError(f"cannot parse a frame number from file name: {name}")
    return FrameId(m.group("stem"), int(m.group("frame")))


def frame_gaps(frames: Sequence[int]) -> np.ndarray:
    """Gaps between consecutive distinct frame numbers after sorting.

    Args:
        frames: Frame numbers from one source (duplicates allowed).

    Returns:
        1-D array of positive gaps, length ``len(unique(frames)) - 1``.
    """
    u = np.unique(np.asarray(frames, dtype=int))
    return np.diff(u)


def gap_percentiles(frames: Sequence[int], q=(5, 25, 50, 75, 90, 95, 99)) -> dict:
    """Summarise the gap distribution to help choose a grouping threshold.

    A clear jump between small gaps (frames of the same tray) and large gaps
    (a new tray or a new capture) suggests a natural threshold.

    Args:
        frames: Frame numbers from one source.
        q: Percentiles to report.

    Returns:
        Dict mapping each percentile to its gap value. Empty if fewer than two
        distinct frames.
    """
    g = frame_gaps(frames)
    if g.size == 0:
        return {}
    return {p: float(np.percentile(g, p)) for p in q}


def group_by_frame_gap(frame_ids: Sequence[FrameId], max_gap: int) -> list[str]:
    """Assign each frame to a group of temporally neighbouring frames.

    Within each source stem, frames are sorted; a new group starts whenever
    the gap to the previous frame is larger than ``max_gap``. Frames with the
    same number (e.g. augmented copies) always share a group.

    Args:
        frame_ids: One `FrameId` per image, in any order.
        max_gap: Largest frame gap still treated as "same tray". Must be >= 0.

    Returns:
        List of group labels (``"<stem>#<k>"``), aligned with ``frame_ids``.

    Raises:
        ValueError: If ``max_gap`` is negative.
    """
    if max_gap < 0:
        raise ValueError("max_gap must be non-negative")

    by_stem: dict[str, list[int]] = defaultdict(list)
    for i, fid in enumerate(frame_ids):
        by_stem[fid.stem].append(i)

    labels = [""] * len(frame_ids)
    for stem, idx in by_stem.items():
        idx = sorted(idx, key=lambda i: frame_ids[i].frame)
        k, prev = 0, None
        for i in idx:
            f = frame_ids[i].frame
            if prev is not None and f - prev > max_gap:
                k += 1
            labels[i] = f"{stem}#{k}"
            prev = f
    return labels


def grouped_split(
    group_profiles: Mapping[str, np.ndarray],
    ratios: Sequence[float] = (0.7, 0.15, 0.15),
    seed: int = 0,
) -> dict[str, int]:
    """Assign whole groups to splits while balancing the class mix.

    Each group has a *profile*: a vector of counts, typically
    ``[n_images, n_background_images, n_boxes_class0, ..., n_boxes_classK]``.
    Groups are placed largest-first (rarest content first) into the split
    whose running totals are furthest below their target share, measured with
    each column weighted by the inverse of its total so rare classes count as
    much as common ones. This is a greedy heuristic; it does not guarantee an
    exact match to the ratios, so always print the resulting per-split counts.

    Args:
        group_profiles: Mapping from group label to a non-negative count
            vector. All vectors must have the same length.
        ratios: Target fraction of each column per split, e.g. train, val,
            test. Must be positive and sum to 1.
        seed: Seed for tie-breaking, so the split is reproducible.

    Returns:
        Mapping from group label to split index (0 = train, 1 = val,
        2 = test for the default ratios).

    Raises:
        ValueError: If ratios are invalid or profiles have inconsistent
            lengths or negative counts.
    """
    ratios = np.asarray(ratios, dtype=float)
    if np.any(ratios <= 0) or not np.isclose(ratios.sum(), 1.0):
        raise ValueError("ratios must be positive and sum to 1")
    if not group_profiles:
        return {}

    names = list(group_profiles)
    P = np.array([np.asarray(group_profiles[g], dtype=float) for g in names])
    if P.ndim != 2:
        raise ValueError("all profiles must have the same length")
    if np.any(P < 0):
        raise ValueError("profile counts must be non-negative")

    totals = P.sum(axis=0)
    weights = np.where(totals > 0, 1.0 / np.maximum(totals, 1e-12), 0.0)
    target = np.outer(ratios, totals)          # (n_splits, n_cols)
    current = np.zeros_like(target)

    rng = np.random.default_rng(seed)
    jitter = rng.random(len(names)) * 1e-9      # reproducible tie-break
    # Rarity score: groups holding rare content are placed first.
    rarity = (P * weights).sum(axis=1) + jitter
    order = np.argsort(-rarity)

    assignment: dict[str, int] = {}
    safe_target = np.maximum(target, 1e-12)
    for gi in order:
        # Fraction of each split's target still unmet, per column. Using the
        # fraction (not the absolute shortfall) stops the large train split
        # from winning every rare group.
        rel_need = (target - current) / safe_target
        # Weight each column by this group's share of that column's total, so
        # the group goes where its own (often rare) content is most needed.
        score = (rel_need * (P[gi] * weights)).sum(axis=1)
        s = int(np.argmax(score))
        current[s] += P[gi]
        assignment[names[gi]] = s
    return assignment


def groups_spanning_splits(groups: Sequence[str], splits: Sequence[int]) -> list[str]:
    """Return groups whose images appear in more than one split.

    Used to audit any split (including Roboflow's original random split). A
    non-empty result means near-duplicate frames are shared between splits.

    Args:
        groups: Group label per image.
        splits: Split index per image, aligned with ``groups``.

    Returns:
        Sorted list of group labels found in two or more splits.
    """
    seen: dict[str, set] = defaultdict(set)
    for g, s in zip(groups, splits):
        seen[g].add(s)
    return sorted(g for g, ss in seen.items() if len(ss) > 1)


# ---------------------------------------------------------------------------
# Independent near-duplicate check (does not use file names)
# ---------------------------------------------------------------------------


def dhash(gray: np.ndarray, hash_size: int = 8) -> np.ndarray:
    """Difference hash of a greyscale image.

    The image is resized to ``(hash_size + 1) x hash_size`` by block
    averaging and each bit records whether a pixel is brighter than its right
    neighbour. Near-identical images produce hashes with a small Hamming
    distance, and the hash is insensitive to small exposure changes.

    Args:
        gray: 2-D array (H, W) of intensities. H must be at least
            ``hash_size`` and W at least ``hash_size + 1``.
        hash_size: Bits per row; the hash has ``hash_size**2`` bits.

    Returns:
        Boolean array of length ``hash_size**2``.

    Raises:
        ValueError: If the input is not 2-D or is too small.
    """
    gray = np.asarray(gray, dtype=float)
    if gray.ndim != 2:
        raise ValueError("dhash expects a 2-D greyscale array")
    h, w = gray.shape
    if h < hash_size or w < hash_size + 1:
        raise ValueError("image too small for the requested hash size")
    rows = np.array_split(np.arange(h), hash_size)
    cols = np.array_split(np.arange(w), hash_size + 1)
    small = np.array([[gray[np.ix_(r, c)].mean() for c in cols] for r in rows])
    return (small[:, 1:] > small[:, :-1]).ravel()


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    """Number of differing bits between two boolean hashes of equal length."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    if a.shape != b.shape:
        raise ValueError("hashes must have the same length")
    return int(np.count_nonzero(a != b))


def cross_split_near_duplicates(
    query_hashes: np.ndarray, reference_hashes: np.ndarray, max_distance: int = 5
) -> np.ndarray:
    """For each query hash, find the closest reference hash distance.

    Typical use: query = test-split hashes, reference = train-split hashes.
    Test images whose nearest train image is within ``max_distance`` bits are
    likely near-duplicates (leakage).

    Args:
        query_hashes: Boolean array (Nq, B).
        reference_hashes: Boolean array (Nr, B).
        max_distance: Distance at or below which a pair counts as a duplicate.

    Returns:
        Boolean array (Nq,), True where a near-duplicate exists. Computed with
        a vectorised XOR, so cost is O(Nq * Nr * B) bit operations.
    """
    q = np.asarray(query_hashes, bool)
    r = np.asarray(reference_hashes, bool)
    if q.size == 0 or r.size == 0:
        return np.zeros(len(q), dtype=bool)
    dist = (q[:, None, :] != r[None, :, :]).sum(axis=2)
    return dist.min(axis=1) <= max_distance
