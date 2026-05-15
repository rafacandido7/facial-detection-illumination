# External Integrations

**Analysis Date:** 2026-05-13

## APIs & External Services

**Model Registry:**
- OpenCV Zoo (GitHub) - Source for the YuNet ONNX model
  - URL: `https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx`
  - Client: Python stdlib `urllib.request.urlretrieve`
  - Auth: None (public URL)
  - Trigger: Auto-downloaded on first run if `models/face_detection_yunet_2023mar.onnx` is missing

**Dataset Source (manual/CLI only):**
- Kaggle Datasets - LFW (Labeled Faces in the Wild) image set
  - Download: `kaggle datasets download jessicali9530/lfw-dataset`
  - Client: Kaggle CLI (not automated in code)
  - Auth: Kaggle API credentials (user-managed, not in codebase)
  - Note: CSV metadata files are committed; images are not versioned

## Data Storage

**Databases:**
- None

**File Storage:**
- Local filesystem only
  - Images: `dataset/lfw/lfw-deepfunneled/lfw-deepfunneled/<person>/<image>.jpg`
  - Custom images: `dataset/proprio/` (not yet committed per README structure)
  - Model weights: `models/face_detection_yunet_2023mar.onnx`
  - LFW metadata CSVs: `dataset/lfw/*.csv`

**Caching:**
- None

## Authentication & Identity

**Auth Provider:**
- Not applicable — offline CLI tool with no user authentication

## Monitoring & Observability

**Error Tracking:**
- None — errors printed to stderr via `sys.stderr` and `sys.exit(1)`

**Logs:**
- stdout only — timing and detection counts printed in structured text format in `src/main.py` (lines 87–90)

## CI/CD & Deployment

**Hosting:**
- Not applicable — local academic project

**CI Pipeline:**
- None detected

## Environment Configuration

**Required env vars:**
- None — fully configured via hardcoded paths and CLI arguments

**Secrets location:**
- No secrets used in the codebase

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None (single HTTP GET to download model, triggered once at startup if model absent)

---

*Integration audit: 2026-05-13*
