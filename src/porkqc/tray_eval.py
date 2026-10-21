"""
Tray-level pass/fail evaluation.

The detector outputs boxes, but the production decision is binary: does the
tray go to the customer or to manual re-inspection? A tray is predicted FAIL
when its highest-confidence detection (any class) is at or above a threshold
``t``. This module scores that decision against ground truth (a tray is truly
defective when its label file contains at least one box).

Engineering priority: a missed defect (false negative) reaches the customer,
while a false alarm (false positive) only costs a manual check. The threshold
is therefore chosen to reach a minimum recall, and the false-alarm rate it
costs is reported.

To avoid an optimistic test score, choose the threshold on the *validation*
split with `select_threshold` and then report test metrics at that fixed
value. Picking the threshold from the test results would let the test data
influence model selection.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def _check(scores, gt):
    scores = np.asarray(scores, dtype=float)
    gt = np.asarray(gt, dtype=bool)
    if scores.shape != gt.shape or scores.ndim != 1:
        raise ValueError("scores and gt must be 1-D arrays of the same length")
    if np.any(~np.isfinite(scores)):
        raise ValueError("scores must be finite")
    return scores, gt


def tray_confusion(scores: Sequence[float], gt: Sequence[bool], threshold: float) -> dict:
    """Tray-level confusion matrix at one threshold.

    Args:
        scores: Highest detection confidence per tray (0 if nothing detected).
        gt: True where the tray is defective.
        threshold: A tray is predicted FAIL when ``score >= threshold``.

    Returns:
        Dict with integer ``tp``, ``fp``, ``fn``, ``tn``. Positive means
        defective / FAIL.

    Raises:
        ValueError: If inputs have mismatched shapes or non-finite scores.
    """
    scores, gt = _check(scores, gt)
    pred = scores >= threshold
    return {
        "tp": int(np.sum(pred & gt)),
        "fp": int(np.sum(pred & ~gt)),
        "fn": int(np.sum(~pred & gt)),
        "tn": int(np.sum(~pred & ~gt)),
    }


def rates(c: dict) -> dict:
    """Recall, precision, false-alarm rate, F1 and accuracy from a confusion dict.

    Undefined ratios (zero denominator) are returned as ``nan`` rather than
    hidden as 0, so an empty class is visible in the results.

    Args:
        c: Dict with ``tp``, ``fp``, ``fn``, ``tn``.

    Returns:
        Dict with ``recall``, ``precision``, ``false_alarm_rate``, ``f1``,
        ``accuracy``.
    """
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]

    def div(a, b):
        return a / b if b else float("nan")

    recall = div(tp, tp + fn)
    precision = div(tp, tp + fp)
    f1 = div(2 * tp, 2 * tp + fp + fn)  # equivalent to 2PR/(P+R), defined when P=0
    return {
        "recall": recall,
        "precision": precision,
        "false_alarm_rate": div(fp, fp + tn),
        "f1": f1,
        "accuracy": div(tp + tn, tp + fp + fn + tn),
    }


def sweep(scores: Sequence[float], gt: Sequence[bool], thresholds: Iterable[float]) -> list[dict]:
    """Confusion counts and rates for each threshold.

    Args:
        scores: Highest detection confidence per tray.
        gt: True where the tray is defective.
        thresholds: Thresholds to evaluate.

    Returns:
        One dict per threshold with ``threshold`` plus the keys of
        `tray_confusion` and `rates`.
    """
    out = []
    for t in thresholds:
        c = tray_confusion(scores, gt, t)
        out.append({"threshold": float(t), **c, **rates(c)})
    return out


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion k/n.

    Preferred over the normal approximation for small ``n`` or proportions
    near 0 or 1, which is the situation here (e.g. 1 false alarm in 12 good
    trays).

    Args:
        k: Number of successes, ``0 <= k <= n``.
        n: Number of trials. ``n = 0`` returns ``(nan, nan)``.
        z: Standard-normal quantile (1.96 for 95 %).

    Returns:
        ``(lower, upper)`` bounds in [0, 1].

    Raises:
        ValueError: If ``k`` is outside ``[0, n]``.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    if not 0 <= k <= n:
        raise ValueError("k must satisfy 0 <= k <= n")
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def select_threshold(
    scores: Sequence[float],
    gt: Sequence[bool],
    min_recall: float = 0.95,
    candidates: Iterable[float] | None = None,
) -> float:
    """Choose the operating threshold on a validation split.

    Returns the *highest* candidate threshold whose recall is at least
    ``min_recall``; a higher threshold means fewer false alarms. If no
    candidate reaches the target, the threshold with the highest recall is
    returned (ties broken by the higher threshold).

    Args:
        scores: Validation-split max confidence per tray.
        gt: Validation-split ground truth.
        min_recall: Required fraction of defective trays caught.
        candidates: Thresholds to consider; defaults to 0.01 to 0.95 in
            steps of 0.01.

    Returns:
        The chosen threshold.

    Raises:
        ValueError: If there are no defective trays (recall undefined).
    """
    scores, gt = _check(scores, gt)
    if not gt.any():
        raise ValueError("validation split has no defective trays")
    if candidates is None:
        candidates = np.round(np.arange(0.01, 0.96, 0.01), 2)
    best_t, best_r = None, -1.0
    ok = []
    for t in candidates:
        r = rates(tray_confusion(scores, gt, t))["recall"]
        if r >= min_recall:
            ok.append(t)
        if r > best_r or (r == best_r and best_t is not None and t > best_t):
            best_t, best_r = t, r
    return float(max(ok)) if ok else float(best_t)


def missed_by_class(
    image_classes: Sequence[Iterable[int]], scores: Sequence[float], threshold: float
) -> dict[int, int]:
    """Count missed defective trays by the defect classes they contain.

    Used in error analysis to see whether misses come from rare classes.
    A tray with two classes is counted under both.

    Args:
        image_classes: For each tray, the ground-truth class ids it contains
            (empty for a good tray).
        scores: Highest detection confidence per tray.
        threshold: Operating threshold.

    Returns:
        Mapping of class id to number of missed trays containing it.
    """
    out: dict[int, int] = {}
    for classes, s in zip(image_classes, scores):
        classes = set(classes)
        if classes and s < threshold:
            for c in classes:
                out[c] = out.get(c, 0) + 1
    return out
