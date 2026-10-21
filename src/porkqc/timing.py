"""
Latency measurement and a simple performance model.

Real-time feasibility is judged against the production-line requirement: if
trays pass the camera at ``r`` trays per second, the full pipeline must take
less than ``1000 / r`` ms per tray on the target hardware.

Measurement rules used here:

* Discard warm-up runs. The first GPU calls include CUDA context creation,
  kernel selection and memory allocation, which can be several times slower
  than steady state.
* Repeat enough times to report a spread (median and 95th percentile), not a
  single number. The 95th percentile matters for a production line because
  the slowest trays set the minimum gap between trays.

The performance model compares the model's arithmetic cost (FLOPs per image,
reported by Ultralytics as GFLOPs) with the time taken, giving achieved
GFLOP/s. Dividing by the device peak shows how far inference is from being
compute-bound; a small fraction suggests the time is dominated by memory
traffic, kernel launch overhead or pre/post-processing instead of arithmetic.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

import numpy as np


@dataclass
class TimingStats:
    """Summary of repeated timings, all in milliseconds."""

    n: int
    mean_ms: float
    std_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float

    def as_dict(self) -> dict:
        """Return the fields as a plain dict (for CSV/JSON output)."""
        return asdict(self)


def summarise(times_ms: Sequence[float]) -> TimingStats:
    """Summarise a list of timings.

    Args:
        times_ms: Individual timings in milliseconds (at least one).

    Returns:
        `TimingStats` for the timings.

    Raises:
        ValueError: If the list is empty.
    """
    t = np.asarray(times_ms, dtype=float)
    if t.size == 0:
        raise ValueError("no timings to summarise")
    return TimingStats(
        n=int(t.size),
        mean_ms=float(t.mean()),
        std_ms=float(t.std(ddof=1)) if t.size > 1 else 0.0,
        median_ms=float(np.median(t)),
        p95_ms=float(np.percentile(t, 95)),
        min_ms=float(t.min()),
        max_ms=float(t.max()),
    )


def benchmark(
    fn: Callable[[Any], Any],
    inputs: Sequence[Any],
    warmup: int = 10,
    repeats: int = 1,
    sync: Callable[[], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> TimingStats:
    """Time ``fn(x)`` for each input after a warm-up phase.

    Args:
        fn: Function to time, called once per input.
        inputs: Inputs to cycle through (e.g. image paths).
        warmup: Number of untimed calls before measuring.
        repeats: Number of passes over ``inputs``.
        sync: Optional function called before each clock read, e.g.
            ``torch.cuda.synchronize``. GPU calls return before the work
            finishes, so without it the timings are too short.
        clock: Timer returning seconds; injectable for testing.

    Returns:
        `TimingStats` over ``len(inputs) * repeats`` timed calls.

    Raises:
        ValueError: If ``inputs`` is empty or counts are negative.
    """
    if len(inputs) == 0:
        raise ValueError("inputs must not be empty")
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be >= 0 and repeats >= 1")
    for i in range(warmup):
        fn(inputs[i % len(inputs)])
    if sync:
        sync()

    times = []
    for _ in range(repeats):
        for x in inputs:
            t0 = clock()
            fn(x)
            if sync:
                sync()
            times.append((clock() - t0) * 1000.0)
    return summarise(times)


def achieved_gflops(gflops_per_image: float, ms_per_image: float) -> float:
    """Achieved arithmetic throughput in GFLOP/s.

    Args:
        gflops_per_image: Model cost per image in GFLOP (Ultralytics reports
            this as "GFLOPs" in the model summary; e.g. 6.4 for YOLO11n at
            640 px).
        ms_per_image: Measured inference time per image in ms.

    Returns:
        GFLOP per second.

    Raises:
        ValueError: If either value is not positive.
    """
    if gflops_per_image <= 0 or ms_per_image <= 0:
        raise ValueError("inputs must be positive")
    return gflops_per_image / (ms_per_image / 1000.0)


def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    """FLOPs per byte of memory traffic (roofline model x-axis).

    Args:
        flops: Floating-point operations.
        bytes_moved: Bytes read and written from device memory.

    Returns:
        FLOP/byte. Compare with the device's ridge point
        (peak FLOP/s divided by memory bandwidth) to judge whether a workload
        is memory- or compute-bound.
    """
    if bytes_moved <= 0:
        raise ValueError("bytes_moved must be positive")
    return flops / bytes_moved


def max_line_rate(p95_ms: float) -> float:
    """Highest tray rate (trays per second) the pipeline can sustain.

    Uses the 95th-percentile time so that 95 % of trays finish in time.

    Args:
        p95_ms: 95th-percentile per-tray processing time in ms.

    Returns:
        Trays per second.
    """
    if p95_ms <= 0:
        raise ValueError("p95_ms must be positive")
    return 1000.0 / p95_ms
