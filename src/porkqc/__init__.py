"""
porkqc: automated packaging-defect inspection for pork rasher trays.

This package holds the reusable, testable parts of the MMA3001 individual
project. The heavy lifting (training and running the YOLO detector) is done by
Ultralytics; this package wraps it and adds the parts that are specific to this
project:

* `porkqc.labels` reads and validates YOLO-format label files.
* `porkqc.grouping` detects near-duplicate video frames and builds a
  leakage-free train/val/test split in which every group of neighbouring
  frames stays in a single split.
* `porkqc.tray_eval` turns per-box detections into a tray-level pass/fail
  decision and evaluates it (confusion matrix, threshold sweep, Wilson
  confidence intervals, threshold selection on the validation split).
* `porkqc.timing` measures inference latency with warm-up and repeats and
  estimates achieved throughput (GFLOP/s) against the hardware peak.
* `porkqc.detect` is a thin, lazily imported wrapper around Ultralytics so the
  rest of the package (and the test suite) runs without a GPU.

Top-level problem
-----------------
Input: an RGB image of a sealed pork rasher tray (any size; resized to
640 px on the long side by the detector).

Output: (1) a list of bounding boxes, each with a defect class and a
confidence in [0, 1]; (2) a tray-level decision, FAIL if any box has
confidence at or above the operating threshold chosen on the validation split,
otherwise PASS.
"""

__version__ = "0.1.0"

#: Class names in the order used by the Roboflow export (YOLO class ids 0..4).
CLASS_NAMES = ("loose-meat", "packaging-error", "twisted-meat", "unsealed", "wrinkle")
