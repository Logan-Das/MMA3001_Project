# AI usage log

The brief requires all material AI use to be disclosed. This file is the
running record; the report's AI reflection summarises it. A row is added
whenever AI is used, with how the output was checked. Dates are DD/MM/YYYY.

| Date | Tool | Task | Level of contribution | How I checked it | Problems found |
|---|---|---|---|---|---|
| 16/10/2026 | Claude (Anthropic) | Project scoping, Colab set-up code, interpreting baseline metrics, tray-level evaluation code | High for code, shared for interpretation | Ran every cell; compared the tray-level counts with the 68 defective / 12 good split in the Ultralytics log | AI first said there were "almost no good trays", which was wrong (12 of 80 are good); corrected from the log |
| 16/10/2026 | Claude | Pointed out possible leakage between splits from the frame numbers in the file names | Suggestion, which I then tested | Measured it myself before changing anything: median distance from a test frame to the nearest training frame was 2.5 frame numbers, and 66 of 80 test frames were within 10 frames of a training frame. The dry run later showed 72 of 184 frame groups (1,012 of 1,317 pictures) spread across the original splits. Confirmed by eye: frames 415 (validation) and 416 (test) show the same tray | The plan assumed I could make a no-augmentation version in Roboflow; I could not (free workspace limit of 1,000 images), so I used Version 4 and removed augmented copies instead |
| 21/10/2026 | Claude | Wrote the `porkqc` package, scripts, tests, notebooks, README and first report draft | High (code and draft text) | Ran the test suite (48 tests pass, `reports/test_report.html`); read each function before using it; compared the split script's counts with the dry-run output | The first grouped-split rule put all rare-class groups in training; a unit test caught it and the scoring rule was changed |
| 22/10/2026 | Claude | Diagnosed the Version 4 data (augmented copies, zoomed regions, frames with two pictures); wrote notebook 03, content-aware de-duplication, recovery and results-upload cells; drafted Sections 5 and 6 of the report and the README results | High (code, analysis and draft text) | Ran notebook 03 and inspected every figure myself; confirmed the rebuilt split was identical to the training split; checked report numbers against the result files in `results/grouped_split_v4/` | (1) An image-hash leakage check was never calibrated: it reported 84–90 % "near-duplicates" that meant nothing, and was dropped in favour of checking by eye. (2) The first de-duplication kept one image per frame number and silently dropped 45 real pictures with their labels; found from my data-check figures and fixed by comparing image content |
| 22/10/2026 | Claude | Interpreted the test, tray-level, benchmark and profiler outputs, the error-case figure and the confidence histogram | Shared: AI proposed explanations, I checked them against the figures | Compared each proposed cause of error with the 19 error images; ran the model myself on random test trays to see its raw output. [Add anything you disagreed with or saw differently] | My own test of the model showed how low its confidence is outside the unsealed class, which led to the histogram analysis |
| 22/10/2026 | Claude | Per-tray score script (merged via pull request #1), presentation deck draft, this log's wording | High (code and draft text) | The script reproduced the evaluated result independently (8 missed, 11 false alarms at t = 0.09); tests passed on the pull request before merging | – |

## Decisions that remained mine

- **Scope:** a tray-level PASS/FAIL decision plus a model comparison, rather than counting rashers or measuring severity.
- **Version 4 rather than Version 1:** Version 4 had more labeled images, it's augmentation was colour only (opposed to V1s rotation, flip and shear) so copies could be safely reduced to the original picture. However the cost is the zoomed view, which you accepted and reported as a limitation.
- **`max_gap = 15`:** 15 sits between the 75th (11) and 90th (19) percentiles of the gaps between sampled frames.The frame strips showed the same tray a few frames apart (415/416, 426/428). A gap of 5 left half of all neighbouring frames in different groups (606 groups), which doesn’t remove leakage. A much larger gap would chain unrelated trays together and leave too few groups to balance the splits.
- **95 % recall target for the threshold:** The two errors cost different amounts. A missed defect, especially an unsealed tray, reaches a customer and is a food-safety and recall risk. A false alarm only costs one manual re-check. Recall comes first, and the false-alarm rate is what's reported as the price.
- **Keeping all five classes:** It keeps the comparison with the baseline fair, since it used the same five classes. The PASS/FAIL decision doesn’t depend on the class, so merging rare classes wouldn’t change the headline result. The rare classes (1–5 test boxes) are reported as too small to interpret, rather than hidden.
- **Choosing YOLO11n over YOLO11s:** At the chosen thresholds, YOLO11s catches only 2 more defective trays (154 vs 152) but raises 6 more false alarms (17 vs 11). It’s the same speed (13.5 ms), so YOLO11s’s extra size buys nothing. It’s smaller (2.6 M vs 9.4 M parameters), which suits line hardware.
- **What to claim:** a PASS covers only the visible part of the tray. The claim reliability for unsealed trays (0 of 72 missed) was very good, but it was not accurate for defect type.

## Notes

- API keys are kept in Colab Secrets and are never committed. 
- Commits written with AI assistance carry a `Co-Authored-By` line.
