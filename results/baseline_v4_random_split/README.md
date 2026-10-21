# Baseline: YOLO11n on Roboflow Version 4 (random split)

These are the first results, kept as the **baseline** for comparison. They were
obtained BEFORE the leakage fix, on Roboflow's random train/val/test split, so
near-duplicate video frames of the same tray may appear in both train and test.
Treat them as an optimistic upper bound, not as the final performance.

| Setting | Value |
|---|---|
| Model | YOLO11n (2.6 M parameters, 6.4 GFLOPs at 640 px) |
| Data | Roboflow V4 export, saturation/exposure augmentation on train |
| Training | 100 epochs, imgsz 640, batch 16, Colab Tesla T4 |
| Software | Ultralytics 8.4.177, Python 3.13, torch 2.11 + CUDA 13.0 |
| Test split | 80 images: 68 defective, 12 good (backgrounds) |

Files:
- `per_class_test.csv`: detection metrics per class on the test split.
- `tray_sweep_test.csv`: tray-level pass/fail at several thresholds.

Notes:
- `wrinkle` has no test instances and is not evaluated.
- `loose-meat` (3 boxes) and `twisted-meat` (4 boxes) are too rare in test to
  support conclusions.
- The 0.15 operating point was chosen by looking at the TEST results, which
  makes those numbers optimistic. The grouped-split runs choose the threshold
  on validation instead (`scripts/evaluate_trays.py`).
- Missed defective trays at t = 0.15: frame_with_red_121, _437, _3371, _3381.
  False alarm: frame_with_red_7471.
- Speed (Ultralytics val log): 5.2 ms pre-process + 8.0 ms inference + 3.3 ms
  post-process per image, i.e. about 16.5 ms per image (no warm-up control).

## Leakage evidence (from notebooks/01_baseline_v4.ipynb)

Frame numbers parsed from the file names, distance from each test frame to the
nearest train frame (train count includes augmented copies):

- test frames: 80, train frames: 3,349
- median nearest-train distance: 2.5 frames
- test frames within 10 frames of a train frame: 66 of 80 (83 %)

Most test trays have a near-identical view of the same tray in training, which
is why the grouped split was built.
