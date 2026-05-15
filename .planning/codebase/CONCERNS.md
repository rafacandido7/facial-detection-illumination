# Codebase Concerns

**Updated:** 2026-05-13
**Severity:** Critical | High | Medium | Low

---

## Critical

### Missing Metrics Pipeline
- No Precision/Recall/F1/IoU computation — the core scientific contribution of the paper
- No ground truth annotation system → cannot validate results
- Without these, the paper's claims cannot be backed by data

### Single-Image Only
- `src/main.py` processes one image at a time
- No batch processing over the LFW dataset
- Cannot produce aggregate stats for the article

### Article Stubs (4 sections TODO)
- `article/main.tex` has 4 incomplete sections with TODO placeholders
- Cannot be submitted in current state

---

## High

### Implementation vs Article Mismatch
- HOG+SVM in code uses `dlib` detector
- Article describes `scikit-image` with specific HOG params
- `scikit-image` is in `requirements.txt` but unused — suggests planned but not implemented approach

### Unused Dependencies (5 of 8)
Declared in `requirements.txt` but not imported anywhere:
- `scikit-image` — planned HOG impl?
- `matplotlib` — planned plotting?
- `pandas` — dataset analysis?
- `Pillow` — image loading?
- `tqdm` — batch progress?

---

## Medium

### Model Re-instantiation on Every Call
- All 3 detector models created fresh each invocation
- Acceptable for single-image CLI; blocking issue for batch mode

### CWD-Relative Paths
- All file paths relative to working directory
- Script breaks if not run from project root

### No Model Integrity Check
- YuNet ONNX model auto-downloaded without checksum verification
- Low risk for academic use; would matter in production

---

## Low

### No Logging
- Errors go to `stderr` via `print()`, no structured logging
- Acceptable for MVP; limits debuggability at scale

### No Tests
- Zero test coverage (see `TESTING.md`)

---

## Summary Priority

1. Add metrics pipeline (Precision/Recall/F1/IoU) — paper cannot be written without this
2. Add batch processing over LFW dataset
3. Reconcile HOG/SVM implementation with article description
4. Complete article stub sections
5. Remove or implement unused dependencies
