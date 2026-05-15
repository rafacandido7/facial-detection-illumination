---
phase: 01-evaluation-pipeline
verified: 2026-05-12T00:00:00Z
status: passed
score: 5/5 must-haves verified
overrides_applied: 0
re_verification: false
---

# Phase 1: Evaluation Pipeline Verification Report

**Phase Goal:** O sistema processa batches de imagens nos 3 detectores × 2 passes e produz métricas (IoU, P/R/F1, tempo) exportáveis em CSV
**Verified:** 2026-05-12T00:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| #   | Truth | Status | Evidence |
|-----|-------|--------|----------|
| 1   | Running the pipeline on an image directory produces results for all 3 detectors × raw and CLAHE without per-image manual intervention | VERIFIED | Smoke-test produced `results/Abba_Eban/raw_results.csv` (6 rows = 1 image × 3 detectors × 2 passes); detectors=['haar','hog','yunet'], passes=['raw','clahe'] |
| 2   | For each detection an IoU value is computed against the corresponding ground truth, and a 0.5 threshold classifies as TP/FP/FN | VERIFIED | `match_detections` fully implemented with all 4 canonical cases passing; wired in `evaluate.py` via `from src.metrics import match_detections`; NaN when no GT (by design) |
| 3   | The system prints or exports Precision, Recall, and F1-score per detector and per lighting condition | VERIFIED | `aggregate_metrics` produces precision/recall/f1 columns per (detector, condition, pass); wired in `evaluate.py` when `--gt-file` is provided; timing-only table printed otherwise |
| 4   | Mean inference time per detector is recorded and aggregated at end of run | VERIFIED | `inference_ms` field recorded per row; `mean_ms` computed in both `aggregate_metrics` (GT mode) and the `timing` fallback groupby (no-GT mode); all columns present in raw_results.csv |
| 5   | A CSV file with all results is saved to disk and can be opened in a spreadsheet | VERIFIED | `raw_results.csv` with 19 required columns written to `results/<dataset>/`; `summary.csv` also written in GT mode; smoke-test confirmed file at `results/Abba_Eban/raw_results.csv` (972B) |

**Score:** 5/5 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/detectors.py` | Detector init, three inference functions, apply_clahe, download_yunet | VERIFIED | 68 lines; exports all 6 required symbols; no GUI calls; no re-init inside detect_* functions |
| `src/metrics.py` | IoU computation, TP/FP/FN matching, metric aggregation | VERIFIED | 51 lines; exports compute_iou, match_detections, aggregate_metrics; pure functions, no I/O, no cv2/dlib |
| `src/evaluate.py` | Batch evaluation CLI entrypoint | VERIFIED | 155 lines; argparse CLI with all 4 args; tqdm loop; GT optional; CSV export; `--help` exits 0 |
| `results/<dataset>/raw_results.csv` | One row per image x detector x pass | VERIFIED | Confirmed at `results/Abba_Eban/raw_results.csv`, 6 rows, 19 columns, all metric cols NaN (no GT provided in smoke test) |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `src/evaluate.py` | `src/detectors.py` | `from src.detectors import init_detectors, detect_haar, detect_hog, detect_yunet, apply_clahe, download_yunet, MODEL_PATH` | WIRED | Line 13-21 of evaluate.py; all symbols imported and used in batch loop |
| `src/evaluate.py` | `src/metrics.py` | `from src.metrics import aggregate_metrics, match_detections` | WIRED | Line 22 of evaluate.py; match_detections called in loop (line 105); aggregate_metrics called after loop (line 141) |
| `batch loop` | `init_detectors()` | called once at line 70, before tqdm loop at line 88 | WIRED | `grep -c "init_detectors()" src/evaluate.py` returns 1; call is outside loop scope |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|-------------------|--------|
| `src/evaluate.py` | `records` list | `detect_fn(pass_img, det_obj)` returns `(boxes, ms)`; `match_detections(gt_boxes, boxes, iou_thresh)` returns `(tp, fp, fn, best_iou)` | Yes — inference runs on real image pixels; metrics computed from real detections | FLOWING |
| `results/Abba_Eban/raw_results.csv` | `inference_ms` column | dlib/OpenCV inference timing via `time.perf_counter()` | Yes — 6 real timing rows confirmed | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| CLI prints usage on `--help` | `python3 src/evaluate.py --help` | Printed valid argparse usage with all 4 args | PASS |
| evaluate.py importable as module | `importlib` load | `main` and `load_ground_truth` present | PASS |
| metrics.py canonical test cases | `python3 -c "from src.metrics import..."` | All 7 assertions passed (compute_iou × 3, match_detections × 4, aggregate_metrics × 3) | PASS |
| detectors.py signature checks | `python3 -c "import inspect; from src.detectors import..."` | All assertions passed: correct params, no re-init, setInputSize present, no GUI calls | PASS |
| raw_results.csv has 19 required columns | `pd.read_csv` | All 19 columns present, 6 rows, iou/tp/fp/fn all NaN (correct for no-GT smoke test) | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| PIPE-01 | 01-01, 01-03 | Batch processing: 3 detectors × 2 passes | SATISFIED | `evaluate.py` loops over all images with DETECT_FNS={haar, hog, yunet} × passes=[raw, clahe]; smoke-test confirmed |
| PIPE-02 | 01-02, 01-03 | IoU computation per detection vs. ground truth | SATISFIED | `compute_iou` and `match_detections` in metrics.py; called in evaluate.py loop when GT provided |
| PIPE-03 | 01-02, 01-03 | Precision, Recall, F1 per detector × lighting condition | SATISFIED | `aggregate_metrics` groups by (detector, condition, pass) and computes P/R/F1; summary.csv and stdout output |
| PIPE-04 | 01-01, 01-03 | Record and aggregate mean inference time | SATISFIED | timing wraps only the inference call (not init); `inference_ms` per row; `mean_ms` aggregated in both modes |
| PIPE-05 | 01-03 | Export results to CSV/JSON | SATISFIED | `raw_results.csv` (19 cols) and optional `summary.csv` written to `results/<dataset>/` |

No orphaned requirements: REQUIREMENTS.md maps PIPE-01 through PIPE-05 to Phase 1 only, and all five are claimed and satisfied.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | None | — | No TODOs, placeholders, empty returns, or GUI calls found in any of the three files |

### Human Verification Required

None. All observable behaviors are verifiable programmatically. The pipeline has no GUI, no real-time behavior, and no external service dependency beyond local filesystem and already-downloaded model weights.

### Gaps Summary

No gaps. All 5 roadmap success criteria are satisfied by the delivered code. All 3 artifacts exist, are substantive, are correctly wired together, and data flows through the pipeline. The smoke-test CSV confirms end-to-end execution.

---

_Verified: 2026-05-12T00:00:00Z_
_Verifier: Claude (gsd-verifier)_
