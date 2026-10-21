"""Tests for porkqc.timing using a fake clock (no GPU needed)."""

import math

import pytest

from porkqc.timing import (
    achieved_gflops,
    arithmetic_intensity,
    benchmark,
    max_line_rate,
    summarise,
)


class FakeClock:
    """Clock that advances by a set amount each time the timed function runs."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_benchmark_excludes_warmup_and_counts_calls():
    clock = FakeClock()
    calls = []

    def fn(x):
        calls.append(x)
        clock.t += 0.010 if len(calls) > 3 else 1.0  # warm-up calls are slow

    stats = benchmark(fn, [1, 2], warmup=3, repeats=2, clock=clock)
    assert len(calls) == 3 + 4
    assert stats.n == 4
    assert math.isclose(stats.median_ms, 10.0)
    assert stats.max_ms < 100  # slow warm-up calls were not timed


def test_benchmark_calls_sync():
    synced = []
    benchmark(lambda x: None, [0], warmup=1, repeats=2, sync=lambda: synced.append(1))
    assert len(synced) == 1 + 2


def test_benchmark_validation():
    with pytest.raises(ValueError):
        benchmark(lambda x: None, [])
    with pytest.raises(ValueError):
        benchmark(lambda x: None, [1], repeats=0)


def test_summarise():
    s = summarise([10, 20, 30, 40])
    assert s.mean_ms == 25 and s.median_ms == 25 and s.min_ms == 10 and s.max_ms == 40
    assert summarise([5]).std_ms == 0.0
    with pytest.raises(ValueError):
        summarise([])


def test_performance_model():
    # YOLO11n: 6.4 GFLOP per image in 8 ms -> 800 GFLOP/s, about 10 % of a T4's 8.1 TFLOP/s.
    assert math.isclose(achieved_gflops(6.4, 8.0), 800.0)
    assert math.isclose(arithmetic_intensity(1e9, 1e8), 10.0)
    assert math.isclose(max_line_rate(20.0), 50.0)
    with pytest.raises(ValueError):
        achieved_gflops(0, 1)
