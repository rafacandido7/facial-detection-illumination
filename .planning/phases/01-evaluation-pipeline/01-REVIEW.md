---
phase: 01-evaluation-pipeline
reviewed: 2026-05-12T00:00:00Z
depth: standard
files_reviewed: 3
files_reviewed_list:
  - src/detectors.py
  - src/metrics.py
  - src/evaluate.py
findings:
  critical: 2
  warning: 5
  info: 3
  total: 10
status: issues_found
---

# Phase 01: Code Review Report

**Reviewed:** 2026-05-12
**Depth:** standard
**Files Reviewed:** 3
**Status:** issues_found

## Summary

Three files implementing a batch face detection evaluation pipeline were reviewed: detector initialization and inference wrappers (`detectors.py`), pure-function metrics computation (`metrics.py`), and the CLI entrypoint (`evaluate.py`).

The most serious issues are a silent GT-lookup failure caused by CWD-relative path resolution (all images silently treated as having no ground truth when the script is not run from the project root), and a correctness bug in the greedy matching algorithm that produces order-dependent TP/FP/FN counts inconsistent with PASCAL VOC / COCO methodology. Both directly affect the validity of the academic evaluation results. Network-download safety and a misleading `best_iou` metric are also notable.

---

## Critical Issues

### CR-01: Silent ground-truth miss due to CWD-relative path resolution

**File:** `src/evaluate.py:96`
**Issue:** `img_path.relative_to(".")` resolves `"."` against the process's current working directory at runtime. When the script is invoked from any directory other than the project root (e.g., `python src/evaluate.py` from inside `src/`), the resulting relative path will not match the keys stored in the GT dict. `gt.get(rel_path)` returns `None`, every image is treated as having no ground truth, and all TP/FP/FN values are silently written as NaN. No warning is emitted. The aggregated CSV then appears valid but contains no accuracy data.

**Fix:** Anchor the relative path computation to the project root, not the CWD:

```python
# Resolve relative to the project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Inside the loop:
rel_path = str(img_path.resolve().relative_to(PROJECT_ROOT))
gt_entry = gt.get(rel_path) if gt else None
```

Also verify that ground-truth JSON keys use the same path separator and root anchor, and add a warning when no GT entries are matched:

```python
if gt is not None and not any(e is not None for e in seen_gt_entries):
    print("WARNING: no ground-truth entries matched any image path. Check GT JSON keys.", file=sys.stderr)
```

---

### CR-02: Greedy matching is order-dependent and diverges from benchmark methodology

**File:** `src/metrics.py:22-33`
**Issue:** `match_detections` iterates GT boxes and takes the *first* detection whose IoU meets the threshold, not the *best-IoU* detection. This means results depend on the order of `det_boxes` (which depends on detector internals). For images with multiple detections, a lower-quality match can displace a higher-quality one, producing different TP/FP/FN counts than PASCAL VOC or COCO evaluation protocols. Academic comparisons across detectors become unreliable.

**Fix:** For each GT box, find the detection with the highest IoU above the threshold, then commit that match:

```python
def match_detections(
    gt_boxes: list,
    det_boxes: list,
    iou_thresh: float = 0.5,
) -> tuple:
    matched_gt: set = set()
    matched_det: set = set()

    # Build all (iou, gi, di) pairs sorted descending by IoU
    pairs = []
    for gi, gt in enumerate(gt_boxes):
        for di, det in enumerate(det_boxes):
            iou_val = compute_iou(gt, det)
            if iou_val >= iou_thresh:
                pairs.append((iou_val, gi, di))
    pairs.sort(key=lambda x: x[0], reverse=True)

    best_matched_iou = 0.0
    for iou_val, gi, di in pairs:
        if gi in matched_gt or di in matched_det:
            continue
        matched_gt.add(gi)
        matched_det.add(di)
        best_matched_iou = max(best_matched_iou, iou_val)

    tp = len(matched_gt)
    fp = len(det_boxes) - len(matched_det)
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn, best_matched_iou
```

---

## Warnings

### WR-01: `best_iou` reports global pairwise maximum, not matched-pair IoU

**File:** `src/metrics.py:23,29`
**Issue:** `best_iou` is initialized outside the GT loop and updated for every pair regardless of whether a match was recorded. On images with multiple GT faces, it reflects the highest IoU between *any* GT-detection pair in the image, which may come from an unmatched pair. The value stored in the CSV (`iou` column) is therefore not the IoU of the actual matched pair and misleads downstream analysis.

**Fix:** Track `best_iou` only over pairs that were successfully matched (handled naturally by the CR-02 fix above). If the current algorithm is kept, reset `best_iou` to `0.0` inside the GT loop and only update it when a match is committed.

---

### WR-02: `download_yunet` has no error handling; partial downloads produce corrupt model

**File:** `src/detectors.py:15-19`
**Issue:** `urllib.request.urlretrieve` is called without a try/except. A network timeout, HTTP 4xx/5xx, or interrupted download writes a partial or empty `.onnx` file to `MODEL_PATH`. The file then exists on disk, so future runs skip the download silently, and OpenCV's `FaceDetectorYN.create` will raise an opaque error (or produce wrong results) when it reads the corrupt file.

**Fix:**

```python
import tempfile, shutil, urllib.error

def download_yunet():
    MODEL_PATH.parent.mkdir(exist_ok=True)
    print("Baixando modelo YuNet (~2 MB)...")
    tmp = MODEL_PATH.with_suffix(".onnx.tmp")
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
        shutil.move(tmp, MODEL_PATH)
        print("Modelo salvo em", MODEL_PATH)
    except (urllib.error.URLError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Falha ao baixar YuNet: {exc}") from exc
```

---

### WR-03: `MODEL_PATH` is a CWD-relative path; silently wrong when invoked from other directories

**File:** `src/detectors.py:12`
**Issue:** `MODEL_PATH = Path("models/face_detection_yunet_2023mar.onnx")` is resolved relative to the CWD at import time. When `init_detectors()` is called from a different working directory, `cv2.FaceDetectorYN.create` receives a path that does not exist. OpenCV may raise a non-obvious error or silently create a broken detector object, crashing only at inference time.

**Fix:** Anchor to the file's own location, consistent with how `evaluate.py` anchors the project root:

```python
_SRC_DIR = Path(__file__).resolve().parent
MODEL_PATH = _SRC_DIR.parent / "models" / "face_detection_yunet_2023mar.onnx"
```

---

### WR-04: Silently skipped images produce no diagnostic output

**File:** `src/evaluate.py:89-91`
**Issue:** When `cv2.imread` returns `None` (corrupt file, unsupported format, permission error), the image is silently skipped with `continue`. In an academic evaluation, skipped images skew recall and detection counts without any indication that data was excluded.

**Fix:**

```python
img = cv2.imread(str(img_path))
if img is None:
    print(f"WARNING: could not read image, skipping: {img_path}", file=sys.stderr)
    continue
```

---

### WR-05: `aggregate_metrics` uses `.sum()` on potentially NaN-containing columns

**File:** `src/metrics.py:43`
**Issue:** In timing-only mode (`gt is None`), `tp`, `fp`, `fn` are written as `float("nan")`. `aggregate_metrics` is only called when `gt is not None` (evaluate.py line 140), so in the current call graph these columns do not contain NaN when the function is invoked. However, `aggregate_metrics` has no internal guard, making it silently produce all-NaN precision/recall/F1 if called directly with a mixed DataFrame, with no error raised.

**Fix:** Add an assertion or explicit NaN check at the top of `aggregate_metrics`:

```python
def aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df[["tp", "fp", "fn"]].isna().any().any():
        raise ValueError("aggregate_metrics received rows with NaN TP/FP/FN; filter timing-only rows before calling.")
    ...
```

---

## Info

### IN-01: `best_det` and `best_gt` stored per row are positionally arbitrary

**File:** `src/evaluate.py:114-115`
**Issue:** `best_det = boxes[0]` and `best_gt = gt_boxes[0]` record only the first element of each list, regardless of which pair was actually matched. For multi-face images, these coordinates are not meaningful as "best" representatives and may confuse anyone reading the raw CSV.

**Fix:** Either remove these coordinate columns from the per-row record (they are not used by `aggregate_metrics`), or store the coordinates of the highest-IoU matched pair returned from `match_detections`.

---

### IN-02: Dataset image scan limited to `.jpg`; other extensions silently excluded

**File:** `src/evaluate.py:77`
**Issue:** `rglob("*.jpg")` excludes `.jpeg`, `.JPG`, `.JPEG`, `.png`, and `.PNG` files without comment. If the dataset contains images with other extensions, they are silently ignored.

**Fix:** Either document the assumption with a comment, or use a multi-extension scan:

```python
EXTENSIONS = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG")
images = sorted(p for ext in EXTENSIONS for p in Path(args.dataset_dir).rglob(ext))
```

---

### IN-03: `open(json_path)` does not specify encoding

**File:** `src/evaluate.py:26`
**Issue:** `open(json_path)` relies on the platform default encoding. On non-UTF-8 systems or when the JSON contains non-ASCII characters (e.g., accented directory names), this can raise `UnicodeDecodeError` or silently misread filenames, causing GT key mismatches.

**Fix:**

```python
with open(json_path, encoding="utf-8") as f:
    data = json.load(f)
```

---

_Reviewed: 2026-05-12_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
