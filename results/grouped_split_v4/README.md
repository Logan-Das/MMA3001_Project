# Results: leakage-free grouped split (Roboflow Version 4)

Produced by `notebooks/02_grouped_pipeline.ipynb` on a Colab Tesla T4,
Ultralytics 8.4.183, PyTorch 2.11 + CUDA 13.0, 22 October 2026.

## Data and split

- Version 4 export, 3,549 files. Colour-augmented copies were removed by content
  (thumbnail correlation >= 0.9), leaving **1,317 distinct pictures from 1,272
  frames** (45 frame numbers hold a second, different zoomed region).
- Frames were grouped with `max_gap = 15` into 184 groups; whole groups were
  assigned to splits. `split_manifest.csv` records every picture's frame, group,
  original split and new split.
- In the original Roboflow split, 72 of the 184 groups (1,012 pictures) were
  spread over two or more splits.

| Split | Pictures | Good trays | loose-meat | packaging-error | twisted-meat | unsealed | wrinkle |
|---|---|---|---|---|---|---|---|
| train | 918 | 178 | 19 | 439 | 21 | 338 | 4 |
| val | 201 | 38 | 5 | 94 | 5 | 72 | 1 |
| test | 198 | 38 | 4 | 95 | 5 | 72 | 1 |

## Files

| File | Contents |
|---|---|
| `yolo11{n,s}_per_class_test.csv` | Precision, recall, AP50, AP50-95 per class on the test split |
| `yolo11{n,s}_tray_summary.json` | Tray-level result at the threshold chosen on validation, Wilson intervals, missed and false-alarm files |
| `yolo11{n,s}_tray_sweep_test.csv` | Tray-level confusion counts and rates at thresholds 0.05 to 0.95 (for the trade-off plot only) |
| `yolo11{n,s}_benchmark_fp32.json`, `_fp16.json` | End-to-end latency: 20 warm-up calls, 594 timed calls, median, 95th percentile, GFLOP/s |
| `yolo11{n,s}_training_log.csv`, `_training_curves.png`, `_train_args.yaml` | Training history and exact settings (100 epochs, 640 px, batch 16, seed 0) |
| `yolo11{n,s}_test_confusion_matrix.png`, `_test_PR_curve.png` | Ultralytics test-split plots |
| `yolo11n_error_cases.png` | The 8 missed and 11 false-alarm test trays, model boxes and labelled boxes (green, dashed) |
| `data_check_*.png` | Data checks from `notebooks/03_data_checks.ipynb`: copies of a frame, Version 1 vs Version 4 framing, a label mismatch, runs of consecutive frames |
| `split_manifest.csv` | The split itself |

## Headline

| | YOLO11n | YOLO11s |
|---|---|---|
| Threshold (validation, recall >= 95 %) | 0.09 | 0.08 |
| Recall, test | 152/160 = 0.95 | 154/160 = 0.96 |
| False-alarm rate, test | 11/38 = 0.29 | 17/38 = 0.45 |
| Mean AP50, unsealed + packaging-error | 0.74 | 0.71 |
| Median / p95 latency, FP32 | 13.5 / 18.2 ms | 13.5 / 17.7 ms |
| Median latency, FP16 | 15.3 ms | 15.6 ms |
