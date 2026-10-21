"""Tests for porkqc.tray_eval: tray-level metrics and threshold selection."""

import math

import numpy as np
import pytest

from porkqc.tray_eval import (
    missed_by_class,
    rates,
    select_threshold,
    sweep,
    tray_confusion,
    wilson_interval,
)

SCORES = [0.9, 0.6, 0.2, 0.05, 0.7, 0.1]
GT = [True, True, True, True, False, False]


def test_confusion_by_hand():
    assert tray_confusion(SCORES, GT, 0.5) == {"tp": 2, "fp": 1, "fn": 2, "tn": 1}
    # The threshold is inclusive: score == threshold counts as FAIL.
    assert tray_confusion([0.5], [True], 0.5)["tp"] == 1


def test_rates_by_hand():
    r = rates({"tp": 2, "fp": 1, "fn": 2, "tn": 1})
    assert r["recall"] == 0.5
    assert math.isclose(r["precision"], 2 / 3)
    assert r["false_alarm_rate"] == 0.5
    assert math.isclose(r["f1"], 2 * (2 / 3) * 0.5 / (2 / 3 + 0.5))
    assert math.isclose(r["accuracy"], 0.5)


def test_rates_report_nan_when_undefined():
    r = rates({"tp": 0, "fp": 0, "fn": 0, "tn": 5})
    assert math.isnan(r["recall"]) and math.isnan(r["precision"])
    assert r["false_alarm_rate"] == 0


def test_reproduces_baseline_sweep_row():
    # Synthetic scores built to match the YOLO11n baseline at t = 0.15:
    # TP 64, FP 1, FN 4, TN 11 -> recall 0.94, precision 0.98, FAR 0.08, F1 0.96.
    scores = np.r_[np.full(64, 0.8), np.full(4, 0.05), np.full(1, 0.5), np.full(11, 0.01)]
    gt = np.r_[np.ones(68, bool), np.zeros(12, bool)]
    c = tray_confusion(scores, gt, 0.15)
    r = rates(c)
    assert c == {"tp": 64, "fp": 1, "fn": 4, "tn": 11}
    assert round(r["recall"], 2) == 0.94 and round(r["precision"], 2) == 0.98
    assert round(r["false_alarm_rate"], 2) == 0.08 and round(r["f1"], 2) == 0.96


def test_sweep_recall_is_non_increasing():
    rng = np.random.default_rng(0)
    s, g = rng.random(200), rng.random(200) > 0.3
    rec = [row["recall"] for row in sweep(s, g, np.linspace(0, 1, 21))]
    assert all(a >= b for a, b in zip(rec, rec[1:]))


def test_input_validation():
    with pytest.raises(ValueError):
        tray_confusion([0.1, 0.2], [True], 0.5)
    with pytest.raises(ValueError):
        tray_confusion([float("nan")], [True], 0.5)


def test_wilson_known_value_and_bounds():
    # 1 false alarm in 12 good trays (baseline): about 1.5 % to 35 %.
    lo, hi = wilson_interval(1, 12)
    assert math.isclose(lo, 0.0149, abs_tol=5e-4) and math.isclose(hi, 0.3539, abs_tol=5e-4)
    assert wilson_interval(0, 10)[0] == 0.0 and wilson_interval(10, 10)[1] == 1.0
    assert all(math.isnan(x) for x in wilson_interval(0, 0))
    with pytest.raises(ValueError):
        wilson_interval(5, 3)


def test_select_threshold_prefers_highest_meeting_recall():
    s = [0.9, 0.8, 0.3, 0.2, 0.25, 0.05]
    g = [True, True, True, True, False, False]
    # Recall 1.0 needs t <= 0.2; recall >= 0.75 allows up to 0.3.
    assert select_threshold(s, g, 1.0) == 0.2
    assert select_threshold(s, g, 0.75) == 0.3


def test_select_threshold_falls_back_to_best_recall():
    s, g = [0.0, 0.5], [True, True]
    # Recall 1.0 is impossible (one score is 0); best recall is 0.5 at t <= 0.5.
    assert select_threshold(s, g, 1.0) == 0.5
    with pytest.raises(ValueError):
        select_threshold([0.1], [False])


def test_missed_by_class():
    classes = [{3}, {1, 2}, set(), {2}]
    scores = [0.9, 0.05, 0.01, 0.1]
    assert missed_by_class(classes, scores, 0.15) == {1: 1, 2: 2}
