# Architecture Map

**Updated:** 2026-05-13
**Pattern:** Single-script academic pipeline (no layers/services)

---

## Pattern

Linear pipeline script. No web server, no database, no modules beyond `src/main.py`. Academic research code comparing face detection algorithms.

## Entry Point

`src/main.py` (~158 lines) — CLI-driven, all logic inline.

## Data Flow

```
CLI args
  → load image from dataset/lfw/
  → apply CLAHE preprocessing (optional second pass)
  → run 3 detectors × 2 passes (raw + CLAHE):
      1. Haar Cascade (OpenCV)
      2. HOG+SVM (dlib)
      3. YuNet CNN (OpenCV DNN, model: models/face_detection_yunet_2023mar.onnx)
  → side-by-side visualization output
  → results/ (empty — outputs not versioned)
```

## Key Components

| Component | Location | Role |
|-----------|----------|------|
| Main pipeline | `src/main.py` | All logic — args, loading, detection, viz |
| YuNet model | `models/face_detection_yunet_2023mar.onnx` | Auto-downloaded on first run |
| Dataset | `dataset/lfw/` | LFW public face dataset (CSV committed, images not versioned) |
| Article | `article/main.tex` | Academic paper (IEEEtran), compiled via `article/Makefile` |
| Results | `results/` | Output dir (empty, not versioned) |

## Abstractions

None. All detector logic, preprocessing, and visualization are inline in `src/main.py`.

## Dependencies

No internal modules. External: OpenCV, dlib, numpy (see `requirements.txt`).

---

*Single-script research pipeline. No architectural complexity.*
