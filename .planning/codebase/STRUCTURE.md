# Directory Structure

**Updated:** 2026-05-13

---

## Layout

```
processamento-de-imagens/
├── src/
│   └── main.py              # Single entry point, all pipeline logic
├── models/
│   └── face_detection_yunet_2023mar.onnx  # Auto-downloaded YuNet model
├── dataset/
│   └── lfw/                 # LFW face dataset
│       └── *.csv            # Metadata committed; images NOT versioned
├── results/                 # Output directory (empty, not versioned)
├── article/
│   ├── main.tex             # Academic paper (IEEEtran template)
│   └── Makefile             # LaTeX build
├── .venv/                   # Python virtual environment
├── requirements.txt         # Python dependencies
├── AV1.pdf                  # Reference/spec document
├── README.md
└── .gitignore
```

## Key Locations

| What | Where |
|------|-------|
| All code | `src/main.py` |
| ML model | `models/face_detection_yunet_2023mar.onnx` |
| Dataset metadata | `dataset/lfw/*.csv` |
| Paper source | `article/main.tex` |
| Build paper | `article/Makefile` → `make` |

## Naming Conventions

- Snake_case for Python (single file, so minimal naming surface)
- No modules/packages — flat `src/` with one file

## What's Not Versioned

- Dataset images (`dataset/lfw/` images excluded)
- Results (`results/`)
- Virtual environment (`.venv/`)
