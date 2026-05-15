---
phase: 01-evaluation-pipeline
fixed_at: 2026-05-12T00:00:00Z
review_path: .planning/phases/01-evaluation-pipeline/01-REVIEW.md
iteration: 1
findings_in_scope: 7
fixed: 7
skipped: 0
status: all_fixed
---

# Phase 01: Code Review Fix Report

**Fixed at:** 2026-05-12
**Source review:** .planning/phases/01-evaluation-pipeline/01-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 7 (2 Critical, 5 Warning)
- Fixed: 7
- Skipped: 0

## Fixed Issues

### CR-01: Silent ground-truth miss due to CWD-relative path resolution

**Files modified:** `src/evaluate.py`
**Commit:** 414768f
**Applied fix:** Added module-level `PROJECT_ROOT = Path(__file__).resolve().parent.parent` constant. Replaced `img_path.relative_to(".")` with `img_path.resolve().relative_to(PROJECT_ROOT)` so GT dict lookups are correct regardless of the CWD at invocation time. Added a post-loop warning to stderr when `gt` is provided but no records contain GT faces (indicating a key-format mismatch).

---

### CR-02: Greedy matching is order-dependent and diverges from benchmark methodology

**Files modified:** `src/metrics.py`
**Commit:** dd3e51f
**Applied fix:** Rewrote `match_detections` to collect all `(iou, gi, di)` pairs that meet the threshold, sort them descending by IoU, then commit highest-quality pairs first via a greedy pass that skips already-matched indices. This is the PASCAL VOC / COCO matching protocol and eliminates dependence on `det_boxes` ordering.

---

### WR-01: `best_iou` reports global pairwise maximum, not matched-pair IoU

**Files modified:** `src/metrics.py`
**Commit:** dd3e51f
**Applied fix:** Resolved naturally by the CR-02 rewrite. The new `best_matched_iou` variable is initialized to `0.0` and updated only when a pair is committed to `matched_gt`/`matched_det`, so it accurately reflects the IoU of the highest-quality actually-matched pair.

---

### WR-02: `download_yunet` has no error handling; partial downloads produce corrupt model

**Files modified:** `src/detectors.py`
**Commit:** 974a392
**Applied fix:** Rewrote `download_yunet` to download into a `.onnx.tmp` temporary file and atomically rename it to `MODEL_PATH` on success. On `urllib.error.URLError` or `OSError`, the partial tmp file is removed via `unlink(missing_ok=True)` and a `RuntimeError` is raised with the original exception chained, preventing a corrupt `.onnx` from persisting.

---

### WR-03: `MODEL_PATH` is a CWD-relative path

**Files modified:** `src/detectors.py`
**Commit:** 974a392
**Applied fix:** Added `_SRC_DIR = Path(__file__).resolve().parent` and derived `MODEL_PATH = _SRC_DIR.parent / "models" / "face_detection_yunet_2023mar.onnx"`, anchoring the model path to the file's own location rather than the process CWD. Also added `import shutil` and `import urllib.error` required by WR-02.

---

### WR-04: Silently skipped images produce no diagnostic output

**Files modified:** `src/evaluate.py`
**Commit:** 75899bd
**Applied fix:** Added `print(f"WARNING: could not read image, skipping: {img_path}", file=sys.stderr)` immediately before the `continue` in the `cv2.imread` None branch, so unreadable/corrupt images are logged to stderr rather than silently excluded from results.

---

### WR-05: `aggregate_metrics` uses `.sum()` on potentially NaN-containing columns

**Files modified:** `src/metrics.py`
**Commit:** ef5e2e6
**Applied fix:** Added an explicit guard at the top of `aggregate_metrics` that calls `df[["tp", "fp", "fn"]].isna().any().any()` and raises `ValueError` with a descriptive message if any NaN values are present, preventing silent all-NaN precision/recall/F1 output when timing-only rows are accidentally passed in.

---

_Fixed: 2026-05-12_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
