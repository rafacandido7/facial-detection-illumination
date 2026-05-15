import argparse
import json
import sys
from pathlib import Path

# Ensure project root is in path when running as script (python src/evaluate.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Anchor relative-path lookups to the project root regardless of CWD
PROJECT_ROOT = Path(__file__).resolve().parent.parent

import cv2
import pandas as pd
from tqdm import tqdm

from src.detectors import (
    MODEL_PATH,
    apply_clahe,
    detect_haar,
    detect_hog,
    detect_yunet,
    download_yunet,
    init_detectors,
)
from src.metrics import aggregate_metrics, match_detections


def load_ground_truth(json_path: str) -> dict:
    with open(json_path) as f:
        data = json.load(f)
    gt = {}
    for entry in data["images"]:
        faces = [
            (face["x1"], face["y1"], face["x2"], face["y2"])
            for face in entry["faces"]
        ]
        gt[entry["file"]] = {"condition": entry["condition"], "faces": faces}
    return gt


def main():
    parser = argparse.ArgumentParser(
        description="Avaliação batch: Haar | HOG+SVM | YuNet com e sem CLAHE"
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Diretório com imagens (percorrido recursivamente)",
    )
    parser.add_argument(
        "--gt-file",
        default=None,
        help="JSON de ground truth (omitir para modo timing-only)",
    )
    parser.add_argument(
        "--out-dir",
        default="results",
        help="Diretório de saída para os CSVs",
    )
    parser.add_argument(
        "--iou-thresh",
        default=0.5,
        type=float,
        help="Limiar IoU para TP/FP/FN",
    )
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        download_yunet()

    gt = load_ground_truth(args.gt_file) if args.gt_file is not None else None

    dets = init_detectors()
    DETECT_FNS = {
        "haar": (detect_haar, dets["haar"]),
        "hog": (detect_hog, dets["hog"]),
        "yunet": (detect_yunet, dets["yunet"]),
    }

    images = sorted(Path(args.dataset_dir).rglob("*.jpg"))
    if len(images) == 0:
        print(f"Erro: nenhuma imagem .jpg encontrada em {args.dataset_dir}", file=sys.stderr)
        sys.exit(1)

    dataset_name = Path(args.dataset_dir).name
    out_dir = Path(args.out_dir) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for img_path in tqdm(images, desc="Evaluating", unit="img"):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"WARNING: could not read image, skipping: {img_path}", file=sys.stderr)
            continue

        img_clahe = apply_clahe(img)
        condition = img_path.parent.name

        rel_path = str(img_path.resolve().relative_to(PROJECT_ROOT))
        gt_entry = gt.get(rel_path) if gt else None
        gt_boxes = gt_entry["faces"] if gt_entry else []

        for pass_name, pass_img in [("raw", img), ("clahe", img_clahe)]:
            for det_name, (detect_fn, det_obj) in DETECT_FNS.items():
                boxes, ms = detect_fn(pass_img, det_obj)

                if gt_entry is not None:
                    tp, fp, fn, best_iou = match_detections(gt_boxes, boxes, args.iou_thresh)
                else:
                    tp, fp, fn, best_iou = (
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                    )

                best_det = boxes[0] if boxes else (None, None, None, None)
                best_gt = gt_boxes[0] if gt_boxes else (None, None, None, None)

                records.append({
                    "image_path":   str(img_path),
                    "condition":    condition,
                    "detector":     det_name,
                    "pass":         pass_name,
                    "gt_x1":        best_gt[0],   "gt_y1": best_gt[1],
                    "gt_x2":        best_gt[2],   "gt_y2": best_gt[3],
                    "det_x1":       best_det[0],  "det_y1": best_det[1],
                    "det_x2":       best_det[2],  "det_y2": best_det[3],
                    "iou":          best_iou,
                    "tp":           tp,
                    "fp":           fp,
                    "fn":           fn,
                    "inference_ms": ms,
                    "n_detections": len(boxes),
                    "n_gt_faces":   len(gt_boxes),
                })

    if gt is not None and records:
        gt_hits = sum(1 for r in records if r.get("n_gt_faces", 0) > 0)
        if gt_hits == 0:
            print(
                "WARNING: no ground-truth entries matched any image path. "
                "Check GT JSON keys vs PROJECT_ROOT-relative paths.",
                file=sys.stderr,
            )

    df = pd.DataFrame(records)
    raw_path = out_dir / "raw_results.csv"
    df.to_csv(raw_path, index=False)
    print(f"\nResultados salvos em {raw_path}")

    if gt is not None:
        summary = aggregate_metrics(df)
        summary_path = out_dir / "summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"Resumo salvo em {summary_path}")
        print("\n" + summary.to_string(index=False))
    else:
        timing = df.groupby(["detector", "pass"])["inference_ms"].mean().reset_index()
        timing.columns = ["detector", "pass", "mean_ms"]
        print("\nTiming (modo sem ground truth):")
        print(timing.to_string(index=False))


if __name__ == "__main__":
    main()
