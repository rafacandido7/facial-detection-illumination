# Technology Stack

**Analysis Date:** 2026-05-13

## Languages

**Primary:**
- Python 3.12.11 - All application logic (`src/main.py`)

**Secondary:**
- LaTeX (IEEEtran template) - Scientific article (`article/main.tex`)

## Runtime

**Environment:**
- CPython 3.12.11 (managed via pyenv)

**Package Manager:**
- pip (venv-based)
- Lockfile: Not present (only `requirements.txt` with version ranges)

## Frameworks

**Core:**
- None (pure Python script — no web framework)

**Build/Dev:**
- Python venv (`.venv/`) - Isolated environment
- LaTeX / Make - Article compilation (`article/Makefile`)

**Testing:**
- Not detected

## Key Dependencies

**Critical:**
- `opencv-python>=4.8.0` - Image I/O, Haar Cascade detector, YuNet (DNN), CLAHE preprocessing, visualization
- `dlib>=19.24.0` - HOG+SVM frontal face detector
- `numpy>=1.24.0` - Array operations, image manipulation

**Supporting:**
- `scikit-image>=0.21.0` - Additional image processing utilities (imported in requirements, not yet used in `src/main.py`)
- `matplotlib>=3.7.0` - Plotting (imported in requirements, not yet used in `src/main.py`)
- `pandas>=2.0.0` - LFW CSV metadata handling
- `Pillow>=10.0.0` - Image loading/conversion utilities
- `tqdm>=4.65.0` - Progress bars for batch processing

## Configuration

**Environment:**
- No `.env` file or environment variables used
- Model path hardcoded: `models/face_detection_yunet_2023mar.onnx`
- Default image path hardcoded: `dataset/lfw/lfw-deepfunneled/lfw-deepfunneled/Abba_Eban/Abba_Eban_0001.jpg`

**Build:**
- `requirements.txt` - Dependency specification with minimum version ranges
- `article/Makefile` - LaTeX build automation

## Platform Requirements

**Development:**
- Python 3.12+ (managed via pyenv)
- `cmake` required to build dlib from source (macOS: `brew install cmake`)
- Virtual environment via `python -m venv .venv`
- YuNet model (~2 MB ONNX file) auto-downloaded from `github.com/opencv/opencv_zoo` on first run

**Production:**
- Single-script CLI tool — no server deployment
- Requires display (GUI window via `cv2.imshow`)
- Model file stored at `models/face_detection_yunet_2023mar.onnx`

---

*Stack analysis: 2026-05-13*
