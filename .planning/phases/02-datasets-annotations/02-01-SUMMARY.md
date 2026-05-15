---
phase: 02-datasets-annotations
plan: "01"
subsystem: dataset-construction
tags: [lfw, ground-truth, haar-detection, image-selection]
dependency_graph:
  requires: []
  provides:
    - scripts/build_lfw_subset.py
    - dataset/lfw_subset/gt.json
    - dataset/lfw_subset/{bright,dark,lateral,overexposed}/
  affects:
    - Phase 03 pipeline execution (consumes gt.json)
tech_stack:
  added: []
  patterns:
    - Haar cascade auto-GT for LFW deepfunneled (consistent with Viola-Jones centering)
    - lfw_attributes.txt score-based image selection (Columbia/CAVE attribute scores)
    - PROJECT_ROOT-relative path keys in gt.json (matches evaluate.py line 100 contract)
key_files:
  created:
    - scripts/build_lfw_subset.py
  modified: []
decisions:
  - "overexposed flash threshold lowered from >1.5 to >1.0 (flash>1.5 yields only 2 candidates in local deepfunneled subset)"
  - "COL_PERSON/COL_IMG indices corrected to 0-based (header comment is 1-based with # prefix; data rows are 0-based)"
  - "dst.relative_to(PROJECT_ROOT) used instead of dst.resolve().relative_to() to avoid symlink path escaping worktree"
metrics:
  duration: "~30 minutes"
  completed: "2026-05-13T13:17:14Z"
  tasks_completed: 1
  tasks_total: 1
  files_created: 1
  files_modified: 0
---

# Phase 02 Plan 01: LFW Subset Construction Summary

**One-liner:** LFW deepfunneled subset of 100 images (25/condition) selected via lfw_attributes.txt scores and auto-annotated with Haar cascade GT bboxes producing gt.json compatible with evaluate.py.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Write scripts/build_lfw_subset.py + run to build dataset | f4ff518 | scripts/build_lfw_subset.py |

## Artifacts Produced

| Artifact | Description |
|----------|-------------|
| `scripts/build_lfw_subset.py` | LFW selection + Haar GT annotation script (153 lines) |
| `dataset/lfw_subset/gt.json` | GT JSON: 100 entries, dataset=lfw_subset, iou_threshold=0.5 |
| `dataset/lfw_subset/bright/` | 25 bright images (soft+outdoor score, soft>1.0 and outdoor>0) |
| `dataset/lfw_subset/dark/` | 25 dark images (harsh<-1.0 and soft<-0.5) |
| `dataset/lfw_subset/overexposed/` | 25 overexposed images (flash>1.0) |
| `dataset/lfw_subset/lateral/` | 25 lateral images (harsh>0.8 and flash<0.5) |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Corrected column indices for lfw_attributes.txt data rows**
- **Found during:** Task 1 execution — script produced 0 candidates with plan-specified constants
- **Issue:** Plan specified `COL_PERSON=1, COL_IMG=2, COL_HARSH=24, COL_FLASH=25, COL_SOFT=26, COL_OUTDOOR=27`. These match the header COMMENT line (which starts with `#\t`), making `#` index 0. But actual data rows have no `#` prefix, so all indices are 1 lower: `COL_PERSON=0, COL_IMG=1, COL_HARSH=23, COL_FLASH=24, COL_SOFT=25, COL_OUTDOOR=26`.
- **Fix:** Corrected all column constants; added explanatory comment in script
- **Files modified:** scripts/build_lfw_subset.py
- **Commit:** f4ff518

**2. [Rule 1 - Bug] Lowered overexposed flash threshold from >1.5 to >1.0**
- **Found during:** Task 1 execution — overexposed condition only produced 2 images with flash>1.5
- **Issue:** Research noted "682 candidates VERIFIED" at flash>1.5 but that was on a different environment. Local deepfunneled subset has only 2 images from people with flash>1.5 scores; flash>1.0 yields 79 valid candidates.
- **Fix:** Changed overexposed filter from `flash > 1.5` to `flash > 1.0`; added explanatory comment
- **Files modified:** scripts/build_lfw_subset.py
- **Commit:** f4ff518

**3. [Rule 3 - Blocking] Symlinked dataset/lfw to main repo; used relative_to() without resolve()**
- **Found during:** Task 1 — worktree doesn't contain large gitignored dataset files
- **Issue:** `dataset/` is in `.gitignore` so each worktree has its own empty dataset dir. LFW images (5749 persons, 13k+ images) only exist in the main repo working tree.
- **Fix:** Created `dataset/lfw` symlink in worktree pointing to main repo's LFW directory. Changed path generation from `dst.resolve().relative_to(PROJECT_ROOT)` to `dst.relative_to(PROJECT_ROOT)` to avoid symlink path escaping the worktree's PROJECT_ROOT.
- **Files modified:** scripts/build_lfw_subset.py
- **Commit:** f4ff518

## Verification Results

All acceptance criteria passed:

```
build_lfw_subset.py static checks passed
gt.json valid: 100 entries
bright: 25 images
dark: 25 images
lateral: 25 images
overexposed: 25 images
Sample GT key: dataset/lfw_subset/bright/Chan_Ho_Park_0001.jpg
```

One Haar miss detected: `Klaus_Schwab_0001.jpg` — fallback bbox (63,63,187,187) applied as designed.

## Known Stubs

None — all gt.json entries have real Haar-detected or fallback bboxes; no placeholder data.

## Threat Flags

None — no new network endpoints, auth paths, or trust boundaries introduced beyond those documented in plan threat model (T-02-01, T-02-02, T-02-03 all accepted/mitigated as planned).

## Self-Check: PASSED

- [x] `scripts/build_lfw_subset.py` exists: FOUND
- [x] `dataset/lfw_subset/gt.json` exists with 100 entries: VERIFIED
- [x] Commit f4ff518 exists: VERIFIED
- [x] All 4 condition directories contain 25 images each: VERIFIED
- [x] GT keys start with `dataset/lfw_subset/` (no leading slash or `./`): VERIFIED
