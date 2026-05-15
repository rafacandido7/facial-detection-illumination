---
phase: 01-evaluation-pipeline
plan: 03
subsystem: evaluate
tags: [cli, batch, csv, tqdm, argparse]
requires: [src/detectors.py, src/metrics.py]
provides: [src/evaluate.py]
affects: []
tech-stack:
  added: [tqdm]
  patterns: [CLI entrypoint, optional GT, batch loop]
key-files:
  created: [src/evaluate.py, results/Abba_Eban/raw_results.csv]
  modified: []
key-decisions:
  - "sys.path.insert(0, project_root) added so script runs as both `python src/evaluate.py` and `python -m src.evaluate`"
  - "init_detectors() called once before tqdm loop — detector objects passed into DETECT_FNS dict"
  - "GT optional: when absent, iou/tp/fp/fn = float('nan'); timing-only table printed instead of summary"
  - "Images skipped with continue (not sys.exit) on cv2.imread None"
requirements-completed: [PIPE-01, PIPE-02, PIPE-03, PIPE-04, PIPE-05]
duration: 2 min
completed: 2026-05-13
---

# Phase 01 Plan 03: Evaluate CLI Summary

Batch evaluation entrypoint wiring detectors.py and metrics.py. Produces raw_results.csv (19 columns) and optional summary.csv. Smoke-tested on Abba_Eban dataset: 6 rows, all IoU/tp/fp/fn NaN (no GT), timing reported per detector×pass.

**Duration:** 2 min | **Start:** 2026-05-13T02:29:02Z | **End:** 2026-05-13T02:30:58Z | **Tasks:** 1 | **Files:** 1

## What Was Built

`src/evaluate.py` with:
- `load_ground_truth(json_path)` — parses GT JSON, returns dict keyed by file path
- `main()` — argparse CLI (`--dataset-dir`, `--gt-file`, `--out-dir`, `--iou-thresh`), batch loop with tqdm, CSV export

Smoke test: `python src/evaluate.py --dataset-dir dataset/lfw/.../Abba_Eban` → 6 rows, 19 cols, all metric cols NaN ✓

## Deviations from Plan

**[Rule 1 - Bug Fix] sys.path insert for script-mode execution** — Found during: Task 1 | `python src/evaluate.py` failed with `ModuleNotFoundError: No module named 'src'` because script mode doesn't set the project root in sys.path | Fix: added `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))` after stdlib imports | Files: src/evaluate.py | Verification: `python src/evaluate.py --help` exits 0 ✓

**Total deviations:** 1 auto-fixed (1 bug fix). **Impact:** None — plan behavior unchanged, all success criteria met.

## Self-Check: PASSED

- `python src/evaluate.py --help` exits 0 ✓
- `results/Abba_Eban/raw_results.csv` exists, 19 columns ✓
- Without --gt-file: iou/tp/fp/fn all NaN ✓
- No imshow/waitKey ✓
- `grep -c "init_detectors()" src/evaluate.py` == 1 ✓
- `git diff src/main.py` → no changes ✓

## Next

Phase 01 complete. Phase 02: ground truth annotation pipeline.
