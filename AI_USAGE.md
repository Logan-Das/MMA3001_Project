# AI usage log

The brief requires all material AI use to be disclosed. This file is the
running record; the report's AI reflection summarises it. Add a row whenever AI
is used, and note how you checked the output.

| Date | Tool | Task | Level of contribution | How I checked it | Problems found |
|---|---|---|---|---|---|
| 2026-10-16 | Claude (Anthropic) | Project scoping, Colab set-up code, interpreting baseline metrics, tray-level evaluation code | High for code, shared for interpretation | Ran every cell; compared the tray table counts with the 68/12 split in the Ultralytics log | AI first said "almost no good trays", which was wrong (12 of 80 are good); it later corrected this from the log |
| 2026-10-16 | Claude | Identified possible frame leakage from the file names | Suggestion, to be confirmed by me with the dry-run counts | *[fill in after running make_grouped_split --dry-run]* | |
| 2026-10-21 | Claude | Wrote the `porkqc` package, scripts, tests, notebook, README and first report draft | High (code and draft text) | *[fill in: e.g. re-ran the tests, read each function, checked the split counts]* | During writing, the first grouped-split heuristic put all rare-class groups in train; the unit test caught it and the scoring rule was changed |
| 2026-10-22 | Claude | Diagnosed the Version 4 data (augmented copies, zoomed regions, frames with two pictures); wrote notebook 03, content-aware de-duplication, the recovery and results-upload cells; drafted Sections 5 and 6 of the report and the README results | High (code, analysis and draft text) | *[fill in]* | AI errors found: (1) a perceptual-hash leakage check whose 5-bit threshold was never calibrated, reported 84-90 % "near-duplicates" that meant nothing, and was dropped; (2) the first de-duplication kept one image per frame number and silently dropped 45 real pictures with their labels, found from my data-check screenshots and fixed |
| 2026-10-22 | Claude | Interpreted the test, tray-level, benchmark and profiler outputs and the error-case figure | Shared: AI proposed explanations, I checked them against the figures | *[fill in: which explanations you agreed or disagreed with]* | |

## Decisions that remained mine

*[Fill in yourself, e.g. choice of scope, the max-gap threshold, the 95 % recall
target, whether to merge rare classes, which model to recommend.]*

## Notes

- API keys are kept in Colab Secrets and are never committed. *[Confirm you
  regenerated the Roboflow key that was pasted into a chat on 2026-10-16.]*
