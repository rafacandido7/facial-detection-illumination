# Phase 1: Evaluation Pipeline - Pattern Map

**Mapped:** 2026-05-12
**Files analyzed:** 3 new files
**Analogs found:** 3 / 3 (all from src/main.py — sole existing source file)

---

## File Classification

| New File | Role | Data Flow | Closest Analog | Match Quality |
|----------|------|-----------|----------------|---------------|
| `src/detectors.py` | service | request-response | `src/main.py` | role-match (extracts + refactors existing detector functions) |
| `src/metrics.py` | utility | transform | `src/main.py` | partial (no metrics logic exists yet; IoU/TP/FP/FN is net-new) |
| `src/evaluate.py` | controller | batch + CRUD | `src/main.py` | role-match (same CLI entrypoint shape; replaces single-image with batch loop) |

---

## Pattern Assignments

### `src/detectors.py` (service, request-response)

**Analog:** `src/main.py`

**Imports pattern** (lines 1-9):
```python
import argparse
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import dlib
import numpy as np
```
For `detectors.py`, keep `cv2`, `dlib`, `time`, `pathlib.Path`. Drop `argparse`, `sys`, `numpy`, `urllib.request` (not needed in this module).

**CLAHE pattern** — copy verbatim from `src/main.py` lines 28-32, rename to `apply_clahe`:
```python
def aplicar_clahe(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
```
Rename to `apply_clahe` (English) for consistency with the new module.

**Haar inference pattern** — refactor from `src/main.py` lines 35-44.
Current (bad — re-inits cascade per call):
```python
def detect_haar(img):
    cascade = cv2.CascadeClassifier(          # <-- init inside fn = WRONG
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t0 = time.perf_counter()
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    elapsed = (time.perf_counter() - t0) * 1000
    boxes = [(x, y, x + w, y + h) for (x, y, w, h) in faces] if len(faces) > 0 else []
    return boxes, elapsed
```
Required change: accept `cascade` as argument (pre-initialized object). Keep the `if len(faces) > 0` guard — it defends against OpenCV version differences where `detectMultiScale` returns `()` vs empty numpy array (see Pitfall 4 in RESEARCH.md).

**HOG inference pattern** — refactor from `src/main.py` lines 47-54.
Current (bad — re-inits dlib per call, ~140 ms overhead):
```python
def detect_hog(img):
    detector = dlib.get_frontal_face_detector()   # <-- init inside fn = WRONG
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t0 = time.perf_counter()
    dets = detector(gray, 1)
    elapsed = (time.perf_counter() - t0) * 1000
    boxes = [(d.left(), d.top(), d.right(), d.bottom()) for d in dets]
    return boxes, elapsed
```
Required change: accept `detector` as argument. Timing must wrap `detector(gray, 1)` only.

**YuNet inference pattern** — refactor from `src/main.py` lines 57-68.
Current (bad — re-inits model per call and passes size at creation):
```python
def detect_yunet(img):
    h, w = img.shape[:2]
    net = cv2.FaceDetectorYN.create(str(MODEL_PATH), "", (w, h))  # <-- init inside fn = WRONG
    t0 = time.perf_counter()
    _, faces = net.detect(img)
    elapsed = (time.perf_counter() - t0) * 1000
    if faces is None:
        return [], elapsed
    boxes = [
        (int(f[0]), int(f[1]), int(f[0] + f[2]), int(f[1] + f[3])) for f in faces
    ]
    return boxes, elapsed
```
Required changes: accept `net` as argument; call `net.setInputSize((w, h))` before each `net.detect()` (required when image dimensions vary across the batch — see Pitfall 2 in RESEARCH.md). `None` guard and box format are correct — keep them.

**Detector init pattern** — net-new, no direct analog in main.py. Use RESEARCH.md Pattern 1:
```python
MODEL_PATH = Path("models/face_detection_yunet_2023mar.onnx")

def init_detectors():
    haar = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    hog = dlib.get_frontal_face_detector()
    yunet = cv2.FaceDetectorYN.create(str(MODEL_PATH), "", (1, 1))
    return {"haar": haar, "hog": hog, "yunet": yunet}
```

**Model download pattern** — copy verbatim from `src/main.py` lines 21-25:
```python
def download_yunet():
    MODEL_PATH.parent.mkdir(exist_ok=True)
    print("Baixando modelo YuNet (~2 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Modelo salvo em", MODEL_PATH)
```
Keep in `detectors.py` so `evaluate.py` can call it before `init_detectors()`.

---

### `src/metrics.py` (utility, transform)

**Analog:** `src/main.py` — no metrics logic exists; this module is entirely net-new.

There is no TP/FP/FN, IoU, or aggregation code in `src/main.py`. The analog for structure only is the module-level function style used throughout `main.py` (plain functions, no classes):

**Module structure pattern** (from `src/main.py` overall):
```python
# Plain functions, no classes, no global state.
# Each function has a single responsibility and returns its output.
def some_fn(arg1, arg2):
    ...
    return result
```

Follow the same convention: all functions in `metrics.py` are pure (no side effects, no I/O), accept positional arguments, return values directly.

**IoU formula** — use RESEARCH.md Pattern 2 (verified against known cases):
```python
def compute_iou(boxA, boxB):
    """Compute IoU between two boxes in (x1, y1, x2, y2) format."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0
```

**Greedy TP/FP/FN matching** — use RESEARCH.md Pattern 2 (verified for 4 canonical cases):
```python
def match_detections(gt_boxes, det_boxes, iou_thresh=0.5):
    """Returns (tp, fp, fn, best_iou)."""
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
            best_iou = max(best_iou, iou_val)
    tp = len(matched_gt)
    fp = len(det_boxes) - len(matched_det)
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn, best_iou
```

**Aggregation** — pandas groupby pattern; no analog in main.py. Use RESEARCH.md Pattern 2:
```python
import pandas as pd

def aggregate_metrics(df):
    grouped = df.groupby(["detector", "condition", "pass"])[["tp", "fp", "fn", "inference_ms"]].sum()
    grouped["precision"] = grouped["tp"] / (grouped["tp"] + grouped["fp"]).clip(lower=1)
    grouped["recall"]   = grouped["tp"] / (grouped["tp"] + grouped["fn"]).clip(lower=1)
    p, r = grouped["precision"], grouped["recall"]
    grouped["f1"]      = 2 * p * r / (p + r).clip(lower=1e-9)
    grouped["mean_ms"] = grouped["inference_ms"] / df.groupby(["detector", "condition", "pass"]).size()
    return grouped.reset_index()
```

---

### `src/evaluate.py` (controller, batch)

**Analog:** `src/main.py`

**CLI / argparse pattern** — copy structure from `src/main.py` lines 103-113:
```python
def main():
    parser = argparse.ArgumentParser(
        description="Comparação Haar | HOG+SVM | YuNet com e sem CLAHE"
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=DEFAULT_IMAGE,
        help="Caminho para a imagem de entrada",
    )
    args = parser.parse_args()
```
Adapt for batch: replace positional `image` with `--dataset`, `--gt` (optional), `--out`, `--iou-thresh`. Keep the same `argparse.ArgumentParser` + `parse_args()` shape.

**Model guard pattern** — copy from `src/main.py` lines 115-116:
```python
    if not MODEL_PATH.exists():
        download_yunet()
```
Place this check before `init_detectors()` in `evaluate.py`.

**Image load + null guard pattern** — copy from `src/main.py` lines 118-121:
```python
    img = cv2.imread(args.image)
    if img is None:
        print(f"Erro: imagem não encontrada — {args.image}", file=sys.stderr)
        sys.exit(1)
```
Adapt for batch: use `continue` instead of `sys.exit(1)` when a single image fails to load mid-batch. Keep the `is None` guard.

**CLAHE application pattern** — copy from `src/main.py` line 123:
```python
    img_clahe = aplicar_clahe(img)
```
Call the renamed `apply_clahe(img)` from `src/detectors.py`.

**Two-pass (raw + CLAHE) loop pattern** — derived from `src/main.py` lines 131-132 (`run_pipeline` called twice):
```python
    panel_orig  = run_pipeline(img,       "Sem CLAHE")
    panel_clahe = run_pipeline(img_clahe, "Com CLAHE")
```
Translate to explicit loop in batch:
```python
for pass_name, pass_img in [("raw", img), ("clahe", img_clahe)]:
    for det_name, (detect_fn, det_obj) in detectors.items():
        boxes, ms = detect_fn(pass_img, det_obj)
        ...
```

**Print / stdout output pattern** — copy banner style from `src/main.py` lines 125-130:
```python
    sep = "=" * 52
    print(sep)
    print("  Detecção Facial — Haar | HOG+SVM | YuNet")
    print(sep)
```
Adapt for summary table output after batch completes.

**Entrypoint guard** — copy from `src/main.py` line 156-157:
```python
if __name__ == "__main__":
    main()
```

**Batch loop skeleton** — net-new pattern (no analog in main.py). Use RESEARCH.md batch loop skeleton:
```python
from tqdm import tqdm
from pathlib import Path
import pandas as pd

images = sorted(Path(dataset_dir).rglob("*.jpg"))
records = []

for img_path in tqdm(images, desc="Evaluating", unit="img"):
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    ...

df = pd.DataFrame(records)
Path(out_dir).mkdir(exist_ok=True)
df.to_csv(f"{out_dir}/raw_results.csv", index=False)
```

---

## Shared Patterns

### Imports
**Source:** `src/main.py` lines 1-9
**Apply to:** All three new files (subset per file)
```python
import argparse
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import dlib
import numpy as np
```
- `detectors.py`: `cv2`, `dlib`, `time`, `pathlib.Path`, `urllib.request`
- `metrics.py`: `pandas` only (no cv2/dlib needed for pure math)
- `evaluate.py`: `argparse`, `sys`, `cv2`, `pathlib.Path`, `pandas`, `tqdm`, plus imports from `src.detectors` and `src.metrics`

### Bounding Box Format Convention
**Source:** `src/main.py` lines 43, 53, 65-67
**Apply to:** All three new files
All boxes are stored and passed as `(x1, y1, x2, y2)` tuples. The only conversion happens at the boundary where raw detector output is normalized:
```python
# Haar: convert (x, y, w, h) → (x1, y1, x2, y2)
boxes = [(x, y, x + w, y + h) for (x, y, w, h) in faces] if len(faces) > 0 else []

# dlib: already provides left/top/right/bottom
boxes = [(d.left(), d.top(), d.right(), d.bottom()) for d in dets]

# YuNet: convert (x, y, w, h) from ONNX output → (x1, y1, x2, y2)
boxes = [(int(f[0]), int(f[1]), int(f[0] + f[2]), int(f[1] + f[3])) for f in faces]
```

### Timing Pattern
**Source:** `src/main.py` lines 40-42, 50-52, 59-61
**Apply to:** `src/detectors.py` (all three detect functions)
```python
t0 = time.perf_counter()
# ... inference call only ...
elapsed = (time.perf_counter() - t0) * 1000   # milliseconds
```
`time.perf_counter()` wraps the inference call only — never wraps initialization.

### Entrypoint Guard
**Source:** `src/main.py` lines 156-157
**Apply to:** `src/evaluate.py`
```python
if __name__ == "__main__":
    main()
```

### No Display Calls in Batch Path
**Source:** `src/main.py` lines 148-153 (negative pattern — do NOT copy)
**Apply to:** `src/evaluate.py` — must never call these:
```python
# NEVER call in evaluate.py (headless batch mode)
cv2.imshow(...)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `src/metrics.py` (IoU + matching functions) | utility | transform | No metrics math exists in codebase; use RESEARCH.md Pattern 2 (verified formulas) |
| `src/metrics.py` (`aggregate_metrics`) | utility | transform | No pandas groupby code exists; use RESEARCH.md Pattern 2 |
| `src/evaluate.py` (batch loop + tqdm) | controller | batch | No batch traversal exists; use RESEARCH.md batch loop skeleton |
| `src/evaluate.py` (GT JSON loading) | controller | file-I/O | No JSON loading exists; use RESEARCH.md Pattern 3 |
| `src/detectors.py` (`init_detectors`) | service | request-response | No pre-init pattern exists in main.py; use RESEARCH.md Pattern 1 |

---

## Metadata

**Analog search scope:** `src/` (sole directory with Python source)
**Files scanned:** 1 (`src/main.py`)
**Pattern extraction date:** 2026-05-12
