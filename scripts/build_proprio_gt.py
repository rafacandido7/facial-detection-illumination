"""Auto-annotate dataset/proprio using HOG (primary) → YuNet → Haar fallback."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from src.detectors import apply_clahe, detect_haar, detect_hog, detect_yunet, init_detectors

ROOT = Path(__file__).resolve().parent.parent
PROPRIO = ROOT / "dataset" / "proprio"
OUT = PROPRIO / "gt.json"

dets = init_detectors()
images = []
failed = []

for img_path in sorted(PROPRIO.rglob("*.jp*g")):
    img = cv2.imread(str(img_path))
    if img is None:
        failed.append(str(img_path.name))
        continue

    condition = img_path.parent.name
    rel_path = str(img_path.relative_to(ROOT))

    # Try HOG first (most conservative, fewer FPs)
    boxes, _ = detect_hog(img, dets["hog"])
    method = "hog"

    # Fallback: YuNet
    if not boxes:
        boxes, _ = detect_yunet(img, dets["yunet"])
        method = "yunet"

    # Fallback: CLAHE + HOG
    if not boxes:
        boxes, _ = detect_hog(apply_clahe(img), dets["hog"])
        method = "clahe+hog"

    # Fallback: Haar — take the detection with highest overlap center
    if not boxes:
        haar_boxes, _ = detect_haar(img, dets["haar"])
        if haar_boxes:
            h, w = img.shape[:2]
            cx, cy = w / 2, h / 2
            boxes = [min(haar_boxes, key=lambda b: abs((b[0]+b[2])/2 - cx) + abs((b[1]+b[3])/2 - cy))]
            method = "haar-center"

    if not boxes:
        print(f"  [SKIP] sem detecção: {img_path.name}")
        failed.append(str(img_path.name))
        continue

    # Use first/only detection as GT
    x1, y1, x2, y2 = boxes[0]
    print(f"  [{method:12s}] {img_path.relative_to(PROPRIO)}  bbox=({x1},{y1},{x2},{y2})")
    images.append({
        "file": rel_path,
        "condition": condition,
        "faces": [{"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}],
    })

gt = {"dataset": "proprio", "iou_threshold": 0.5, "images": images}
OUT.write_text(json.dumps(gt, indent=2, ensure_ascii=False))
print(f"\nGT salvo: {OUT} ({len(images)} imagens, {len(failed)} falhas)")
if failed:
    print(f"Sem GT: {failed}")
