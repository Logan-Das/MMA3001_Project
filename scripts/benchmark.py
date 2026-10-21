"""
Measure end-to-end inference latency per tray and the achieved GFLOP/s.

Times the full `model.predict` call (pre-processing, inference and NMS), which
is what the production line experiences, after a warm-up phase.

Example:
    python scripts/benchmark.py --weights runs/yolo11n_grouped/weights/best.pt \
        --images /content/pork_grouped/test/images --gflops 6.4 \
        --peak-tflops 8.1 --out results/yolo11n_grouped
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from porkqc.detect import load_model  # noqa: E402
from porkqc.labels import list_images  # noqa: E402
from porkqc.timing import achieved_gflops, benchmark, max_line_rate  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--gflops", type=float, required=True, help="Model GFLOPs from the Ultralytics summary")
    ap.add_argument("--peak-tflops", type=float, default=8.1, help="Device FP32 peak (T4: 8.1)")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--half", action="store_true", help="FP16 inference")
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    model = load_model(args.weights)
    imgs = [str(p) for p in list_images(args.images)]
    try:
        import torch
        sync = torch.cuda.synchronize if torch.cuda.is_available() else None
        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except ImportError:
        sync, device = None, "cpu"

    stats = benchmark(lambda p: model.predict(p, conf=0.01, verbose=False, half=args.half),
                      imgs, warmup=args.warmup, repeats=args.repeats, sync=sync)
    gfs = achieved_gflops(args.gflops, stats.median_ms)
    result = {
        "device": device, "half": args.half, **stats.as_dict(),
        "achieved_gflop_per_s": gfs,
        "fraction_of_peak": gfs / (args.peak_tflops * 1000),
        "max_trays_per_s_at_p95": max_line_rate(stats.p95_ms),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        name = "benchmark_fp16.json" if args.half else "benchmark_fp32.json"
        (Path(args.out) / name).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
