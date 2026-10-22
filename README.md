# MMA3001 Project: pork rasher packaging-defect inspection

MMA3001 Numerical Methods and Machine Learning, individual project (2026).

[![Open the pipeline in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Logan-Das/MMA3001_Project/blob/main/notebooks/02_grouped_pipeline.ipynb)
![tests](https://github.com/Logan-Das/MMA3001_Project/actions/workflows/ci.yml/badge.svg)

A computer-vision pipeline that inspects images of sealed pork rasher trays,
localises packaging and placement defects with a YOLO11 object detector, and
turns the detections into a tray-level **PASS / FAIL** decision for a
production line. The project's main contribution is the evaluation: a
leakage-free data split, a pass/fail threshold chosen on validation data, and
latency measured against a line-rate requirement.

![pipeline](docs/pipeline.svg)

<img width="640" alt="Example tray from the dataset" src="docs/dataset_example.jpg" />

## Problem in one paragraph

Manual inspection of sealed trays is slow and inconsistent, and a defective
tray (e.g. an unsealed film) that reaches a customer is a food-safety and
recall risk. Given one RGB image of a tray, the system outputs (1) boxes with a
defect class and confidence, and (2) a FAIL decision if any box reaches the
operating threshold. Missing a defect is treated as worse than a false alarm,
which only costs a manual re-check.

## Inputs and outputs

| | Description |
|---|---|
| **Input** | RGB image of a tray on the line, JPEG/PNG, any size (the dataset is 720 × 540, a zoomed region of the overhead camera view; the detector resizes to 640 px). |
| **Output 1** | Boxes `(class, confidence ∈ [0,1], x, y, w, h)`. Classes: `loose-meat`, `packaging-error`, `twisted-meat`, `unsealed`, `wrinkle`. |
| **Output 2** | Tray decision: FAIL if max confidence ≥ *t*, else PASS. *t* is chosen on the validation split to reach ≥ 95 % recall (0.09 for the final YOLO11n). |
| **Labels** | YOLO text files, one box per line, normalised coordinates. An empty or missing file means a good tray. |
| **Invalid input** | Malformed label lines raise `LabelFormatError` naming the file and line. File names without a frame number stop the split script rather than being guessed. Images from other camera set-ups are out of scope (see Limitations). |

## Repository layout

```
src/porkqc/          Python package (documented, unit-tested)
  labels.py          read and validate YOLO labels
  grouping.py        content-aware de-duplication, frame grouping, leakage-free split
  tray_eval.py       pass/fail metrics, threshold selection, Wilson intervals
  timing.py          latency benchmark and GFLOP/s performance model
  detect.py          thin wrapper around Ultralytics (imported lazily)
scripts/             command-line entry points used by the notebook
  make_grouped_split.py, train.py, evaluate_trays.py, benchmark.py,
  plot_tray_scores.py
notebooks/
  01_baseline_v4.ipynb      first experiment: YOLO11n on the Roboflow V4 random split
  02_grouped_pipeline.ipynb  leakage-free experiments, run in order on a Colab T4
  03_data_checks.ipynb      checks on copies, cropping and labels (CPU only)
  colab_git_sync.ipynb      clone / commit / push this repo from Colab
tests/               pytest unit and end-to-end tests (no GPU needed)
results/             metrics from each experiment (CSV/JSON)
docs/                project brief, figures, generated HTML API docs (docs/api)
reports/             generated pytest HTML report
```

## Reproducing the results

1. **Tests and docs (any computer, CPU only)**

   ```bash
   pip install -r requirements.txt
   pip install -e .
   make test     # runs pytest and writes reports/test_report.html
   make docs     # writes docs/api/index.html from the docstrings
   ```

2. **Experiments (Google Colab, T4 GPU).** Open
   `notebooks/02_grouped_pipeline.ipynb` in Colab, add your Roboflow API key as a
   Colab Secret named `ROBOFLOW_API_KEY` (and `GIT_TOKEN` if the repo is private), and run the cells in order. Outputs
   are written to Google Drive.

Random seeds are fixed (`seed=0`, `deterministic=True`) and every split
decision is recorded in `split_manifest.csv`. GPU training is not bit-for-bit
deterministic, so metrics can vary slightly between runs.

## Data

Dataset: [Pork Rasher Error (Packaging) on Roboflow Universe](https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging),
about 1,955 original 720 × 540 frames from video of a packing line, with
bounding-box labels. The images are **not** stored in this repository; the
notebook downloads them. Check the dataset page for its licence.

## Results

All numbers are on the **test split**, used once after every choice (training
stop, threshold, model) was made on validation. Tray-level intervals are 95 %
Wilson intervals.

| | YOLO11n baseline | **YOLO11n** | YOLO11s |
|---|---|---|---|
| Split | Roboflow random (leaky) | grouped, leakage-free | grouped, leakage-free |
| Test trays (defective / good) | 80 (68 / 12) | 198 (160 / 38) | 198 (160 / 38) |
| Threshold *t* (chosen on) | 0.15 (test) | 0.09 (validation) | 0.08 (validation) |
| Recall: defective trays caught | 0.94 | **0.95** (0.90 to 0.97) | 0.96 (0.92 to 0.98) |
| False-alarm rate: good trays failed | 0.08 (1/12) | **0.29** (11/38; 0.17 to 0.45) | 0.45 (17/38; 0.30 to 0.60) |
| Unsealed trays missed | not measured | **0 of 72** | 1 of 72 |
| AP50 unsealed / packaging-error | 0.84 / 0.46 | **0.93 / 0.55** | 0.91 / 0.50 |
| Latency, median / 95th pct (T4, FP32, batch 1) | not controlled | **13.5 / 18.2 ms** | 13.5 / 17.7 ms |

Key findings:

- **The first result was optimistic.** In Roboflow's random split, 72 of 184
  groups of neighbouring video frames (1,012 of 1,317 pictures) were spread across
  splits. On the leakage-free split with the threshold fixed on validation, recall
  holds at 95 % but the false-alarm rate is about 1 in 3, not 1 in 12.
- **YOLO11n is the recommended model.** It catches as many defective trays as
  YOLO11s, or more, at every false-alarm rate, with a quarter of the parameters.
- **Speed is not the constraint.** About 18 ms per tray against a 300 to 667 ms
  budget for a 90 to 200 packs/min line. Model size and FP16 barely change the
  time, because kernel launches and image decoding dominate at batch size 1.
- **Errors** are mostly near misses just under the threshold, small defects at the
  edge of the zoomed image, and packaging-error boxes that look like features
  labelled inconsistently elsewhere (see `results/grouped_split_v4/yolo11n_error_cases.png`).

Files: `results/grouped_split_v4/` (this work) and
`results/baseline_v4_random_split/` (first attempt). Model weights are kept on
Google Drive, not in the repository.

## Limitations

- **Each image shows part of the scene.** Version 4 images are zoomed regions
  (about a quarter) of the camera view, so trays are often cut off. A PASS means no
  defect was seen in the visible region.
- One camera set-up and lighting condition; performance on other lines is unknown.
- Rare classes (`loose-meat`, `twisted-meat`, `wrinkle`: 4, 5 and 1 test boxes)
  are too small for per-class conclusions.
- 38 good test trays still give a wide interval on the false-alarm rate.
- Labels are the only ground truth, and some look inconsistent between
  neighbouring frames of the same tray.

## AI declaration

I acknowledge the use of AI to assist in the generation, modification, and development of code within this project. AI tools were used to support coding, debugging, and documentation, while all final decisions, implementation, testing, and responsibility for the project's content remain my own.

A dated log of what AI was used for and how it was checked is in [AI_USAGE.md](AI_USAGE.md). Commits written with AI assistance carry a `Co-Authored-By` line.

## Licence

Code: MIT (see `LICENSE`). Dataset and pretrained weights keep their own licences.
