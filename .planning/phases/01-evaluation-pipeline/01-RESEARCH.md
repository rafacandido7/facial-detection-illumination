# Phase 1: Evaluation Pipeline - Research

**Researched:** 2026-05-12
**Domain:** Python batch image processing, object detection metrics, IoU / P/R/F1 computation
**Confidence:** HIGH

---

## Summary

Phase 1 transforms `src/main.py` (a single-image, interactive visualization tool) into a
headless batch evaluation pipeline. The core work is: (1) refactor detector functions to
accept pre-initialized detector objects so timing is fair, (2) implement IoU and
TP/FP/FN matching, (3) loop over a directory of images loading ground truth from a
per-dataset JSON file, and (4) aggregate and export results as CSV.

The existing stack (OpenCV 4.13, dlib 20.0.1, numpy 2.4, pandas 3.0, tqdm 4.67, scipy)
is sufficient for all requirements with zero new dependencies. The most important
architectural insight is that `src/main.py` re-initializes detectors per image call
— this inflates dlib timing by ~140 ms per image and must be fixed before any timing
results are meaningful.

Ground truth will be stored as a single JSON file per dataset (`annotations.json`) with
a flat list of image records. This format is trivial to write by hand, compatible with
labelme exports, and readable with stdlib `json` — no PASCAL VOC XML or COCO tooling
needed for a 20-image dataset.

**Primary recommendation:** Create `src/evaluate.py` as the new batch entrypoint; keep
`src/main.py` intact for interactive visualization. Extract shared detector logic into
`src/detectors.py` (detector init + inference, separated). Call `evaluate.py` from the
CLI as `python src/evaluate.py --dataset dataset/proprio --gt annotations.json --out results/`.

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PIPE-01 | System processes image batches with 3 detectors × 2 passes (raw + CLAHE) | Batch loop with tqdm; detector pre-init pattern; pathlib.glob for image discovery |
| PIPE-02 | Compute IoU between each detection and corresponding ground truth | Standard intersection-over-union formula verified; greedy matching strategy selected |
| PIPE-03 | Compute Precision, Recall, F1 per detector × illumination condition (IoU ≥ 0.5) | Micro-average over summed TP/FP/FN per group; pandas groupby for aggregation |
| PIPE-04 | Record and aggregate average inference time per detector | time.perf_counter() around inference only; mean across images per detector |
| PIPE-05 | Export results as CSV/JSON readable in a spreadsheet | pandas DataFrame → CSV; flat row-per-image-per-detector-per-pass schema |
</phase_requirements>

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Detector initialization | `src/detectors.py` module | — | Shared between main.py and evaluate.py; init once, pass as argument |
| Batch image loading | `src/evaluate.py` | — | Only evaluate.py needs batch traversal |
| Ground truth loading | `src/evaluate.py` | — | JSON parsing at startup, keyed by relative image path |
| IoU / TP/FP/FN matching | `src/metrics.py` module | — | Pure functions, easily unit-testable independently |
| CLAHE preprocessing | Already in `src/main.py` → extract to `src/detectors.py` | — | Shared across passes in both main.py and evaluate.py |
| Results aggregation | `src/evaluate.py` (pandas) | — | groupby detector × condition for micro-averaged P/R/F1 |
| CSV export | `src/evaluate.py` (pandas) | — | DataFrame.to_csv() |
| Interactive visualization | `src/main.py` (unchanged) | — | Keep existing behavior; evaluate.py is headless |

---

## Standard Stack

### Core (already installed, no new deps needed)

| Library | Installed Version | Purpose | Why Standard |
|---------|-------------------|---------|--------------|
| opencv-python | 4.13.0 [VERIFIED: pip show] | Haar Cascade, YuNet, CLAHE, image I/O | Already used; FaceDetectorYN available |
| dlib | 20.0.1 [VERIFIED: pip show] | HOG+SVM face detection | Already used; get_frontal_face_detector() |
| numpy | 2.4.4 [VERIFIED: pip show] | Array ops for IoU math | Zero-overhead, already present |
| pandas | 3.0.2 [VERIFIED: pip show] | Results DataFrame, CSV export, groupby aggregation | Already in requirements.txt |
| tqdm | 4.67.3 [VERIFIED: pip show] | Progress bar over image batch | Already in requirements.txt |
| scipy | Available [VERIFIED: import test] | Hungarian matching (scipy.optimize.linear_sum_assignment) | stdlib-like, already in venv |
| json | stdlib [VERIFIED: import test] | Ground truth annotation file parsing | No extra dependency |
| pathlib | stdlib | Image directory traversal via rglob | Clean API |
| time | stdlib | Inference timing with perf_counter() | Nanosecond resolution |

**No new packages to install.** [VERIFIED: all confirmed via pip show and import tests]

---

## Architecture Patterns

### System Architecture Diagram

```
CLI args (--dataset, --gt, --out, --iou-thresh)
        |
        v
  evaluate.py main()
        |
        +---> load_ground_truth(json_path)
        |         returns: dict[image_path -> list[bbox]]
        |
        +---> init_detectors()               # src/detectors.py
        |         returns: {haar: obj, hog: obj, yunet: obj}
        |
        v
  for image in tqdm(glob(dataset/**/*.jpg)):
        |
        +---> img = cv2.imread(image)
        +---> img_clahe = apply_clahe(img)   # src/detectors.py
        |
        +---> for pass in [('raw', img), ('clahe', img_clahe)]:
        |         for detector_name, detect_fn in detectors.items():
        |             |
        |             +---> boxes, ms = detect_fn(pass_img)
        |             +---> gt_boxes = ground_truth.get(image, [])
        |             +---> iou_val = best_iou(boxes, gt_boxes)  # src/metrics.py
        |             +---> tp, fp, fn = match(gt_boxes, boxes)  # src/metrics.py
        |             +---> append row to records[]
        |
        v
  df = pd.DataFrame(records)
  df.to_csv(out/raw_results.csv)
        |
        v
  summary = aggregate(df)   # groupby detector × condition
  summary.to_csv(out/summary.csv)
        |
        v
  print summary table to stdout
```

### Recommended Project Structure

```
src/
├── main.py          # Existing interactive visualizer (DO NOT BREAK)
├── evaluate.py      # New: batch evaluation entrypoint
├── detectors.py     # New: detector init + inference functions (extracted from main.py)
└── metrics.py       # New: iou(), match_detections(), aggregate_metrics()
annotations/
├── proprio.json     # Ground truth for dataset/proprio (created in Phase 2)
└── lfw_subset.json  # Ground truth for LFW subset (created in Phase 2)
results/
├── raw_results.csv  # One row per image × detector × pass
└── summary.csv      # Aggregated P/R/F1 + mean timing per detector × condition
```

### Pattern 1: Detector Pre-initialization (CRITICAL for fair timing)

**What:** Initialize detector objects once before the batch loop; pass them as arguments.
**When to use:** Always. Current `main.py` re-initializes dlib per call (140 ms overhead each).

```python
# Source: [VERIFIED: measured dlib init = 139.8ms, YuNet init = 3.1ms]

# src/detectors.py
import cv2
import dlib
from pathlib import Path

MODEL_PATH = Path("models/face_detection_yunet_2023mar.onnx")

def init_detectors():
    haar = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    hog = dlib.get_frontal_face_detector()
    yunet = cv2.FaceDetectorYN.create(str(MODEL_PATH), "", (1, 1))
    return {"haar": haar, "hog": hog, "yunet": yunet}


def detect_haar(img, cascade):
    """Time ONLY inference, not init."""
    import time
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t0 = time.perf_counter()
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    boxes = [(x, y, x+w, y+h) for x, y, w, h in faces] if len(faces) > 0 else []
    return boxes, elapsed_ms


def detect_hog(img, detector):
    """dlib HOG — init already done, time inference only."""
    import time
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t0 = time.perf_counter()
    dets = detector(gray, 1)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return [(d.left(), d.top(), d.right(), d.bottom()) for d in dets], elapsed_ms


def detect_yunet(img, net):
    """YuNet — resize input to match image dims."""
    import time
    h, w = img.shape[:2]
    net.setInputSize((w, h))
    t0 = time.perf_counter()
    _, faces = net.detect(img)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    if faces is None:
        return [], elapsed_ms
    return [(int(f[0]), int(f[1]), int(f[0]+f[2]), int(f[1]+f[3])) for f in faces], elapsed_ms


def apply_clahe(img):
    """Identical to main.py aplicar_clahe — extract here to share."""
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
```

### Pattern 2: IoU and Greedy TP/FP/FN Matching

**What:** Standard intersection-over-union for bounding boxes; greedy matching for TP/FP/FN.
**When to use:** Single GT face per image (our primary case). Greedy is equivalent to Hungarian for 1 GT face and is simpler.

```python
# Source: [VERIFIED: formula tested with known cases; results confirmed correct]

# src/metrics.py

def compute_iou(boxA, boxB):
    """Compute IoU between two boxes (x1,y1,x2,y2 format)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def match_detections(gt_boxes, det_boxes, iou_thresh=0.5):
    """
    Greedy matching: for each GT box, find first unmatched detection with IoU >= thresh.
    Returns (tp, fp, fn, best_iou).
    best_iou: IoU of best matched pair (or 0 if no match).
    """
    matched_gt = set()
    matched_det = set()
    best_iou = 0.0

    for gi, gt in enumerate(gt_boxes):
        for di, det in enumerate(det_boxes):
            if di in matched_det:
                continue
            iou_val = compute_iou(gt, det)
            if iou_val >= iou_thresh:
                matched_gt.add(gi)
                matched_det.add(di)
                best_iou = max(best_iou, iou_val)
                break
            best_iou = max(best_iou, iou_val)  # track even sub-threshold

    tp = len(matched_gt)
    fp = len(det_boxes) - len(matched_det)
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn, best_iou


def aggregate_metrics(df):
    """
    Compute micro-averaged Precision/Recall/F1 per (detector, condition, pass).
    Micro-average: sum TPs/FPs/FNs first, then divide. Standard for detection.
    """
    grouped = df.groupby(["detector", "condition", "pass"])[["tp", "fp", "fn", "inference_ms"]].sum()
    grouped["precision"] = grouped["tp"] / (grouped["tp"] + grouped["fp"]).clip(lower=1)
    grouped["recall"] = grouped["tp"] / (grouped["tp"] + grouped["fn"]).clip(lower=1)
    p = grouped["precision"]
    r = grouped["recall"]
    grouped["f1"] = 2 * p * r / (p + r).clip(lower=1e-9)
    grouped["mean_ms"] = grouped["inference_ms"] / df.groupby(["detector", "condition", "pass"]).size()
    return grouped.reset_index()
```

### Pattern 3: Ground Truth JSON Format

**What:** Flat JSON file mapping relative image paths to lists of bounding boxes.
**When to use:** Per dataset, written once manually (or via labelme export in Phase 2).

```json
{
  "dataset": "proprio",
  "iou_threshold": 0.5,
  "images": [
    {
      "file": "dataset/proprio/boa_iluminacao/img001.jpg",
      "condition": "boa_iluminacao",
      "faces": [
        {"x1": 120, "y1": 80, "x2": 280, "y2": 300}
      ]
    }
  ]
}
```

Loading pattern:
```python
import json

def load_ground_truth(json_path):
    """Returns dict: relative_path -> {'condition': str, 'faces': list[tuple]}"""
    with open(json_path) as f:
        data = json.load(f)
    gt = {}
    for entry in data["images"]:
        boxes = [(face["x1"], face["y1"], face["x2"], face["y2"])
                 for face in entry["faces"]]
        gt[entry["file"]] = {"condition": entry["condition"], "faces": boxes}
    return gt
```

### Pattern 4: CSV Results Schema

**What:** One row per (image × detector × pass). Includes raw detection coords, IoU, TP/FP/FN, timing.
**Why this structure:** Maximum flexibility — can re-aggregate any way in Phase 3; nothing is lost.

```
image_path | condition | detector | pass | gt_x1 | gt_y1 | gt_x2 | gt_y2 |
det_x1 | det_y1 | det_x2 | det_y2 | iou | tp | fp | fn | inference_ms |
n_detections | n_gt_faces
```

For images with no ground truth (pipeline smoke-test before Phase 2), `gt_*` columns are `NaN`.

### Pattern 5: YuNet setInputSize (required for batch)

**What:** YuNet requires `net.setInputSize((w, h))` before each call if image size varies.
**Critical:** Current `main.py` passes size at creation time — correct for single image but must
call `setInputSize` per image in the batch loop if images have different dimensions.

```python
# In batch loop, per image:
h, w = img.shape[:2]
net.setInputSize((w, h))  # BEFORE net.detect(img)
_, faces = net.detect(img)
```

### Anti-Patterns to Avoid

- **Re-initializing detectors per image:** Inflates dlib timing by ~140 ms/image.
  Fix: init once, pass object as argument.
- **Timing model load in YuNet:** `cv2.FaceDetectorYN.create()` loads ONNX model.
  This is not "inference time". Time only `net.detect()`.
- **Storing timing in detector function when detector is not pre-initialized:**
  The old `main.py` pattern times `dlib.get_frontal_face_detector()` + inference together.
  This makes HOG look 10–30× slower than it actually is for inference alone.
- **Using display window (`cv2.imshow`) in batch mode:** `evaluate.py` must be headless.
  Never call `cv2.imshow` or `cv2.waitKey` in the evaluation path.
- **Writing GT bounding boxes in (x, y, w, h) format:** OpenCV uses (x,y,w,h);
  our internal format and dlib both use (x1,y1,x2,y2). Pick one and convert at the boundary.
  Recommendation: store as x1,y1,x2,y2 everywhere internally; convert Haar output at load.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Progress bar over image batch | Custom print loop | `tqdm` | Already installed; handles ETA, rate, nested |
| CSV export | Manual file.write() | `pandas.DataFrame.to_csv()` | Handles quoting, encoding, NaN; already installed |
| Grouped aggregation | Manual dict loops | `pandas.groupby()` | Correct handling of empty groups, multi-index |
| Hungarian matching (multi-face) | Custom algorithm | `scipy.optimize.linear_sum_assignment` | Optimal matching in O(n³); scipy already in venv |
| Image path globbing | os.walk() | `pathlib.Path.rglob('*.jpg')` | Cleaner, cross-platform |

**Key insight:** For single-face images (LFW and our dataset), greedy matching is identical
to Hungarian. Only implement Hungarian if multi-face images appear — scipy makes it a
3-line addition.

---

## Common Pitfalls

### Pitfall 1: Timing Includes Detector Initialization
**What goes wrong:** dlib reports ~150 ms per image instead of ~10 ms; HOG appears pathologically slow.
**Why it happens:** Current `main.py` calls `dlib.get_frontal_face_detector()` inside `detect_hog()`.
Initialization is ~140 ms [VERIFIED: measured]; inference is ~10-30 ms.
**How to avoid:** Extract `init_detectors()` called once before the batch loop; inject objects.
**Warning signs:** dlib timing > 100 ms on small images.

### Pitfall 2: YuNet Input Size Mismatch
**What goes wrong:** `cv2.FaceDetectorYN.detect()` returns garbage or crashes if `setInputSize`
was not called for the current image size.
**Why it happens:** YuNet's ONNX backend requires explicit input dimensions.
**How to avoid:** Call `net.setInputSize((w, h))` at the start of each image's inference.
**Warning signs:** YuNet detects faces in wrong positions; FP count very high.

### Pitfall 3: Ground Truth Path Mismatch
**What goes wrong:** `gt.get(image_path)` returns `None` for every image.
**Why it happens:** GT JSON stores paths as `"dataset/proprio/foo/img.jpg"` but the glob
returns absolute paths or paths from a different working directory.
**How to avoid:** Normalize all paths to the same relative root (project root). Use
`Path(image_path).relative_to(project_root)` and store as string keys.
**Warning signs:** All images show `n_gt_faces = 0`.

### Pitfall 4: Haar detectMultiScale Returns Empty Array (not empty list)
**What goes wrong:** `len(faces)` works, but iterating `for x, y, w, h in faces` fails when
`faces` is a numpy array of shape `(0,)`.
**Why it happens:** `detectMultiScale` returns `()` (empty tuple) or a numpy array depending
on OpenCV version. Current `main.py` already handles this with `if len(faces) > 0`.
**How to avoid:** Keep the existing guard: `[(x,y,x+w,y+h) for x,y,w,h in faces] if len(faces) > 0 else []`
**Warning signs:** TypeError during Haar box unpacking.

### Pitfall 5: Pipeline Requires GT to Run (blocks smoke-testing)
**What goes wrong:** Can't run `evaluate.py` until Phase 2 annotations exist.
**Why it happens:** Hard-coded `--gt required` CLI arg without a skip option.
**How to avoid:** Make `--gt` optional; if absent, skip IoU/TP/FP/FN columns (set to NaN)
and only record timing + detection count. Allows smoke-testing the batch loop now.
**Warning signs:** Can't test pipeline before dataset annotation.

---

## Code Examples

### Batch Loop Skeleton

```python
# Source: [VERIFIED: tqdm API, pandas 3.x, pathlib stdlib]

import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.detectors import init_detectors, detect_haar, detect_hog, detect_yunet, apply_clahe
from src.metrics import match_detections

def run_evaluation(dataset_dir, gt=None, iou_thresh=0.5, out_dir="results/"):
    detectors = init_detectors()
    images = sorted(Path(dataset_dir).rglob("*.jpg"))
    records = []

    for img_path in tqdm(images, desc="Evaluating", unit="img"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_clahe = apply_clahe(img)

        # Derive condition from parent directory name
        condition = img_path.parent.name

        gt_entry = gt.get(str(img_path.relative_to(".")), None) if gt else None
        gt_boxes = gt_entry["faces"] if gt_entry else []

        for pass_name, pass_img in [("raw", img), ("clahe", img_clahe)]:
            for det_name, (detect_fn, det_obj) in detectors.items():
                boxes, ms = detect_fn(pass_img, det_obj)
                tp, fp, fn, best_iou = match_detections(gt_boxes, boxes, iou_thresh)
                best_det = boxes[0] if boxes else (None, None, None, None)
                best_gt  = gt_boxes[0] if gt_boxes else (None, None, None, None)
                records.append({
                    "image_path": str(img_path),
                    "condition": condition,
                    "detector": det_name,
                    "pass": pass_name,
                    "gt_x1": best_gt[0], "gt_y1": best_gt[1],
                    "gt_x2": best_gt[2], "gt_y2": best_gt[3],
                    "det_x1": best_det[0], "det_y1": best_det[1],
                    "det_x2": best_det[2], "det_y2": best_det[3],
                    "iou": best_iou,
                    "tp": tp, "fp": fp, "fn": fn,
                    "inference_ms": ms,
                    "n_detections": len(boxes),
                    "n_gt_faces": len(gt_boxes),
                })

    df = pd.DataFrame(records)
    Path(out_dir).mkdir(exist_ok=True)
    df.to_csv(f"{out_dir}/raw_results.csv", index=False)
    return df
```

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| opencv-python | Haar, YuNet, CLAHE, imread | Yes | 4.13.0 | — |
| dlib | HOG+SVM detection | Yes | 20.0.1 | — |
| numpy | IoU math | Yes | 2.4.4 | — |
| pandas | DataFrame, CSV export | Yes | 3.0.2 | — |
| tqdm | Progress bar | Yes | 4.67.3 | — |
| scipy | Hungarian matching (multi-face) | Yes | available | Greedy (sufficient for 1-face images) |
| json | GT file parsing | Yes | stdlib | — |
| pathlib | Image discovery | Yes | stdlib | — |
| time.perf_counter | Inference timing | Yes | stdlib | — |
| YuNet ONNX model | YuNet inference | Yes | 2023mar | Download via existing download_yunet() |
| dataset/proprio | Smoke-test with real images | No | — | Use any LFW image for pipeline smoke-test |
| annotations JSON | IoU metrics | No | — | Run without --gt (timing-only mode) |

**Missing dependencies with no fallback:** None.

**Missing dependencies with fallback:**
- `dataset/proprio` — not yet captured (Phase 2). Use LFW images to verify batch loop mechanics.
- `annotations.json` — not yet created (Phase 2). Run pipeline with `--gt` omitted to verify timing and detection count columns.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Each image in both datasets contains exactly 1 face | Pitfalls, Metrics | Multi-face images need Hungarian matching and per-match row schema |
| A2 | Illumination condition is derivable from parent directory name | Batch Loop | If structure differs, condition parsing breaks; would need it in GT JSON |
| A3 | LFW subset for Phase 2 will have ground truth annotations available externally | STATE.md | If no public LFW bbox annotations found, Phase 2 requires manual annotation of LFW subset |
| A4 | Greedy matching is sufficient (vs. Hungarian) given single-face assumption | Metrics | Acceptable: scipy is already available if multi-face case appears |

---

## Open Questions (RESOLVED)

1. **Best IoU for multi-detection images**
   - What we know: For 1 GT face + N detections, greedy picks best detection first.
   - What's unclear: Should CSV store best-matched detection's bbox, or all detections?
   - RESOLVED: Store only the best-matched detection's bbox per row; log `n_detections` so FP count is recoverable. Implemented in Plan 01-03.

2. **LFW ground truth source**
   - What we know: LFW has no official bounding box annotations; the included CSVs are pair/identity metadata only.
   - What's unclear: Will a public LFW bbox dataset be usable (e.g., WIDER FACE subset, or LFW-a annotations)?
   - RESOLVED: Deferred to Phase 2. Phase 1 pipeline only needs the JSON schema defined — annotation is Phase 2 work.

3. **Output path convention**
   - What we know: `results/` directory exists and is gitignored.
   - RESOLVED: Use `results/{dataset_name}/` as output dir so LFW and próprio results don't collide. Implemented in Plan 01-03.

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| Per-call detector init (main.py) | Pre-init once, inject object | dlib timing accurate: ~10-30ms not ~150ms |
| Interactive window output | Headless CSV output | Compatible with CI/scripting |
| Single image entrypoint | Batch directory traversal | PIPE-01 satisfied |

---

## Sources

### Primary (HIGH confidence)
- [VERIFIED: pip show / import] — All library versions confirmed in running environment
- [VERIFIED: Python measurement] — dlib init 139.8 ms, YuNet init 3.1 ms, Haar init 8.0 ms
- [VERIFIED: IoU formula test] — intersection/union formula confirmed against known cases (0.0, 0.333, 1.0)
- [VERIFIED: Hungarian test] — scipy.optimize.linear_sum_assignment correctly assigns max-IoU pair
- [VERIFIED: greedy match test] — TP/FP/FN counts verified for 4 canonical cases
- [VERIFIED: tqdm] — tqdm.tqdm works in CLI batch context

### Secondary (MEDIUM confidence)
- [ASSUMED] — Micro-average (sum TP/FP/FN per group, then compute metrics) is the standard aggregation strategy for face detection benchmarks. This is common practice but not verified against a specific benchmark spec for this project's scope.

### Tertiary (LOW confidence)
- None.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all library versions verified via pip show and import tests in the running venv
- Architecture: HIGH — patterns derived from code inspection of existing main.py + verified timing measurements
- Pitfalls: HIGH — timing inflation verified by measurement; YuNet setInputSize from code reading; others from code analysis
- IoU/metrics formulas: HIGH — verified with explicit test cases

**Research date:** 2026-05-12
**Valid until:** 2026-06-12 (stable libraries; project deadline is 2026-05-14 anyway)
