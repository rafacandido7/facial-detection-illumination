---
phase: 01-evaluation-pipeline
plan: 02
subsystem: metrics
tags: [metrics, iou, precision, recall, f1, pandas]
requires: []
provides: [src/metrics.py]
affects: [src/evaluate.py]
tech-stack:
  added: []
  patterns: [pure functions, pandas groupby micro-average]
key-files:
  created: [src/metrics.py]
  modified: []
key-decisions:
  - "Greedy matching: outer loop over GT, inner over detections — first match above thresh wins, break inner"
  - "best_iou tracked for all pairs including sub-threshold, ensures non-zero value even for misses"
  - "aggregate_metrics uses pandas .clip(lower=1) for precision/recall to avoid division by zero"
requirements-completed: [PIPE-02, PIPE-03]
duration: 3 min
completed: 2026-05-13
---

# Phase 01 Plan 02: Metrics Module Summary

Net-new pure-function metrics module. IoU formula, greedy TP/FP/FN matching at configurable threshold, and pandas micro-averaged aggregation per (detector, condition, pass).

**Duration:** 3 min | **Start:** 2026-05-13T02:26:18Z | **End:** 2026-05-13T02:29:02Z | **Tasks:** 1 | **Files:** 1

## What Was Built

`src/metrics.py` with:
- `compute_iou(boxA, boxB)` — exact intersection-over-union formula, returns 0.0 on zero-area
- `match_detections(gt_boxes, det_boxes, iou_thresh=0.5)` — greedy matching, returns (tp, fp, fn, best_iou)
- `aggregate_metrics(df)` — groupby (detector, condition, pass), micro-avg P/R/F1, mean_ms

## Deviations from Plan

None - plan executed exactly as written.

## Self-Check: PASSED

- All 4 match_detections canonical cases pass ✓
- All 3 compute_iou canonical cases pass (1.0, 0.0, 25/175) ✓
- aggregate_metrics produces correct precision=0.5, recall=0.5, mean_ms=11.0 on test df ✓
- No print, no cv2/dlib imports ✓

## Next

Ready for 01-03 (src/evaluate.py — Wave 2).
