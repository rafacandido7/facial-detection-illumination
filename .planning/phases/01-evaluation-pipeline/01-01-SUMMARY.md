---
phase: 01-evaluation-pipeline
plan: 01
subsystem: detectors
tags: [detectors, cv2, dlib, yunet, performance]
requires: [src/main.py]
provides: [src/detectors.py]
affects: [src/evaluate.py]
tech-stack:
  added: []
  patterns: [pre-initialized detector, shared module]
key-files:
  created: [src/detectors.py]
  modified: []
key-decisions:
  - "Detectors initialized once in init_detectors() dict — callers pass the pre-built object, never re-create it"
  - "detect_yunet calls net.setInputSize((w,h)) before t0 to exclude resize from timing"
  - "apply_clahe renamed from aplicar_clahe for English consistency with evaluate.py API"
requirements-completed: [PIPE-01, PIPE-04]
duration: 15 min
completed: 2026-05-13
---

# Phase 01 Plan 01: Detector Module Summary

Pre-initialized detector module extracted from `src/main.py`. All three detector functions now accept a pre-built detector object instead of constructing one per call — eliminating the 139.8 ms dlib re-init overhead per image.

**Duration:** 15 min | **Start:** 2026-05-13T02:10:59Z | **End:** 2026-05-13T02:26:18Z | **Tasks:** 1 | **Files:** 1

## What Was Built

`src/detectors.py` with:
- `init_detectors()` → `{"haar": cascade, "hog": dlib_detector, "yunet": yunet_net}`
- `detect_haar(img, cascade)`, `detect_hog(img, detector)`, `detect_yunet(img, net)` — timing wraps inference only
- `apply_clahe(img)` — renamed from `aplicar_clahe`
- `download_yunet()` — verbatim from main.py

## Deviations from Plan

None - plan executed exactly as written.

## Self-Check: PASSED

- `from src.detectors import init_detectors, detect_haar, detect_hog, detect_yunet, apply_clahe, download_yunet` ✓
- No internal detector init inside any detect_* function ✓
- `detect_yunet` calls `setInputSize` before `perf_counter` ✓
- `init_detectors()` returns dict with keys {"haar", "hog", "yunet"} ✓
- No imshow/waitKey in file ✓
- `git diff src/main.py` → no changes ✓

## Next

Ready for 01-02 (src/metrics.py) and 01-03 (src/evaluate.py — Wave 2).
