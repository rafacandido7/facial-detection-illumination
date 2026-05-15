"""
Build LFW subset: select 25 images/condition from lfw-deepfunneled via lfw_attributes.txt,
copy to dataset/lfw_subset/, auto-annotate GT bboxes via Haar cascade.

Paper methodology note (Section III / Section V):
  GT bounding boxes for the LFW subset were generated via Haar cascade
  (haarcascade_frontalface_default.xml), consistent with the lfw-deepfunneled creation
  procedure which used Viola-Jones (Haar) centering. Haar F1 scores on this subset
  therefore represent an upper bound relative to the other detectors.
"""
import cv2
import json
import shutil
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ATTR_URL = "https://www.cs.columbia.edu/CAVE/databases/pubfig/download/lfw_attributes.txt"
ATTR_LOCAL = PROJECT_ROOT / "dataset/lfw/lfw_attributes.txt"
LFW_DIR = PROJECT_ROOT / "dataset/lfw/lfw-deepfunneled/lfw-deepfunneled"
OUT_DIR = PROJECT_ROOT / "dataset/lfw_subset"
FALLBACK_BBOX = (63, 63, 187, 187)
N_PER_CONDITION = 25

# Column indices for data rows (0-based Python indices).
# lfw_attributes.txt header comment line is: #\tperson\timagenum\t...
# The '#' prefix shifts header indices by 1 vs actual data rows.
# Data row: [0]=person_name, [1]=imagenum, [2..74]=attributes
# Header col 24 (Harsh Lighting) -> data index 23; col 25 (Flash) -> 24; etc.
COL_PERSON = 0
COL_IMG = 1
COL_HARSH = 23
COL_FLASH = 24
COL_SOFT = 25
COL_OUTDOOR = 26

CONDITIONS = {
    "bright": (
        lambda r: float(r[COL_SOFT]) + float(r[COL_OUTDOOR]),
        lambda r: float(r[COL_SOFT]) > 1.0 and float(r[COL_OUTDOOR]) > 0,
    ),
    "dark": (
        lambda r: -(float(r[COL_HARSH]) + float(r[COL_SOFT])),
        lambda r: float(r[COL_HARSH]) < -1.0 and float(r[COL_SOFT]) < -0.5,
    ),
    "overexposed": (
        lambda r: float(r[COL_FLASH]),
        # Threshold lowered from 1.5 to 1.0: flash>1.5 yields only ~2 candidates in
        # the available deepfunneled subset; flash>1.0 yields 79+ candidates.
        lambda r: float(r[COL_FLASH]) > 1.0,
    ),
    "lateral": (
        lambda r: float(r[COL_HARSH]),
        lambda r: float(r[COL_HARSH]) > 0.8 and float(r[COL_FLASH]) < 0.5,
    ),
}


def main():
    # Step 1: Download attributes file if missing
    if not ATTR_LOCAL.exists():
        print("Downloading lfw_attributes.txt...")
        ATTR_LOCAL.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(ATTR_URL, ATTR_LOCAL)
        print(f"Downloaded: {ATTR_LOCAL}")

    # Step 2: Parse attribute file
    with open(ATTR_LOCAL) as f:
        lines = f.readlines()
    data_lines = [
        l.split("\t") for l in lines if not l.startswith("#") and l.strip()
    ][1:]  # skip attribute header row (first non-comment line is column names)

    # Step 3: Init Haar cascade ONCE
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Step 4: Initialize gt_entries list
    gt_entries = []

    # Step 5: For each condition, select and copy images, detect bboxes
    for cond, (score_fn, filter_fn) in CONDITIONS.items():
        cond_dir = OUT_DIR / cond
        cond_dir.mkdir(parents=True, exist_ok=True)

        # Build candidates: filter rows, build src path, check exists
        candidates = []
        for row in data_lines:
            try:
                if not filter_fn(row):
                    continue
                name = row[COL_PERSON].replace(" ", "_").strip()
                imgnum = int(row[COL_IMG])
                src = LFW_DIR / name / f"{name}_{imgnum:04d}.jpg"
                if src.exists():
                    candidates.append((score_fn(row), src))
            except (ValueError, IndexError):
                continue

        # Sort descending, take top N_PER_CONDITION
        candidates.sort(reverse=True)
        selected = candidates[:N_PER_CONDITION]
        if len(selected) < N_PER_CONDITION:
            print(f"WARNING: {cond} only has {len(selected)} candidates (need {N_PER_CONDITION})")

        for _, src in selected:
            dst = cond_dir / src.name
            shutil.copy2(src, dst)

            # Haar-detect on copied image
            img = cv2.imread(str(dst))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                x, y, w, h = faces[0]
                bbox = {"x1": int(x), "y1": int(y), "x2": int(x + w), "y2": int(y + h)}
            else:
                print(f"WARNING: no face detected in {dst.name}, using fallback bbox")
                x1, y1, x2, y2 = FALLBACK_BBOX
                bbox = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

            # Use PROJECT_ROOT-relative path directly (not via resolve()) to avoid
            # following symlinks (dataset/lfw is a symlink on some setups).
            # dst is under OUT_DIR which is always PROJECT_ROOT / "dataset/lfw_subset".
            rel = str(dst.relative_to(PROJECT_ROOT))
            gt_entries.append({"file": rel, "condition": cond, "faces": [bbox]})

        print(f"{cond}: {len(selected)} images copied")

    # Step 6: Write gt.json
    gt = {"dataset": "lfw_subset", "iou_threshold": 0.5, "images": gt_entries}
    gt_path = OUT_DIR / "gt.json"
    with open(gt_path, "w") as f:
        json.dump(gt, f, indent=2)
    print(f"GT written: {gt_path} ({len(gt_entries)} images)")

    # Step 7: Self-check — verify no absolute or ./ paths in gt.json
    if gt_entries:
        sample_key = gt_entries[0]["file"]
        assert not sample_key.startswith("/"), f"FAIL: absolute path in GT: {sample_key}"
        assert not sample_key.startswith("./"), f"FAIL: ./ prefix in GT: {sample_key}"
        assert sample_key.startswith("dataset/lfw_subset/"), (
            f"FAIL: expected 'dataset/lfw_subset/' prefix, got: {sample_key}"
        )
        print(f"Sample GT key: {sample_key}")
    else:
        print("WARNING: no gt_entries produced — check lfw_attributes.txt and LFW_DIR")


if __name__ == "__main__":
    main()
