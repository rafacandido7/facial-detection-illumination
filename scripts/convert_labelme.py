"""
Convert labelme JSON annotations to unified gt.json for dataset/proprio/.

Usage:
  1. Capture ~5 images per condition into dataset/proprio/{bright,dark,lateral,overexposed}/
  2. Annotate each image with: labelme --nodata
     Draw one rectangle per face; save; labelme writes a .json beside the image.
  3. Run: python scripts/convert_labelme.py
  4. Verify: dataset/proprio/gt.json created with one entry per annotated image.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROPRIO_DIR = PROJECT_ROOT / "dataset/proprio"


def parse_labelme_json(json_path: Path) -> dict | None:
    """Parse a single labelme JSON file and return a gt.json image entry, or None.

    Returns None if no rectangle annotation is found (and prints a WARNING).
    """
    with open(json_path) as f:
        data = json.load(f)

    faces = []
    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "rectangle":
            continue
        pts = shape["points"]  # [[x1,y1],[x2,y2]] — corner order may vary
        faces.append(
            {
                "x1": int(min(pts[0][0], pts[1][0])),
                "y1": int(min(pts[0][1], pts[1][1])),
                "x2": int(max(pts[0][0], pts[1][0])),
                "y2": int(max(pts[0][1], pts[1][1])),
            }
        )

    if not faces:
        print(f"WARNING: no rectangle annotation in {json_path}")
        return None

    # CRITICAL: imagePath is filename-only relative to JSON file's directory.
    # Always join with json_path.parent before resolving to an absolute path.
    img_file = (json_path.parent / data["imagePath"]).resolve()
    rel = str(img_file.relative_to(PROJECT_ROOT))
    condition = json_path.parent.name  # bright / dark / lateral / overexposed

    return {"file": rel, "condition": condition, "faces": faces}


def main() -> None:
    # Glob all JSON files under dataset/proprio/, but skip gt.json itself.
    jsons = sorted(PROPRIO_DIR.rglob("*.json"))
    jsons = [j for j in jsons if j.name != "gt.json"]

    entries = [e for j in jsons if (e := parse_labelme_json(j)) is not None]

    if len(entries) == 0:
        print(
            "WARNING: no annotations found — run labelme first.\n"
            "Steps:\n"
            "  1. pip install labelme\n"
            "  2. labelme --nodata dataset/proprio/bright/  (repeat for each condition)\n"
            "  3. Draw a rectangle around each face; File > Save.\n"
            "  4. Re-run: python scripts/convert_labelme.py"
        )
        sys.exit(1)

    gt = {"dataset": "proprio", "iou_threshold": 0.5, "images": entries}
    out = PROPRIO_DIR / "gt.json"

    with open(out, "w") as f:
        json.dump(gt, f, indent=2)

    print(f"GT written: {out} ({len(entries)} images)")

    # Self-check: verify file keys are PROJECT_ROOT-relative (no leading / or ./)
    sample = entries[0]["file"]
    print(f"Sample file key: {sample!r}")
    assert not sample.startswith("/"), f"FATAL: absolute path in gt.json key: {sample!r}"
    assert not sample.startswith("./"), f"FATAL: ./ prefix in gt.json key: {sample!r}"
    assert sample.startswith("dataset/proprio/"), (
        f"FATAL: unexpected prefix in gt.json key: {sample!r}"
    )
    print("Self-check passed: file keys are PROJECT_ROOT-relative.")


if __name__ == "__main__":
    main()
