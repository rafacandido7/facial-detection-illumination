import pandas as pd


def compute_iou(boxA: tuple, boxB: tuple) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def match_detections(
    gt_boxes: list,
    det_boxes: list,
    iou_thresh: float = 0.5,
) -> tuple:
    matched_gt: set = set()
    matched_det: set = set()

    # Build all (iou, gi, di) pairs that meet the threshold, sorted desc by IoU.
    # Committing the highest-IoU pair first matches PASCAL VOC / COCO methodology
    # and eliminates order-dependence on det_boxes ordering.
    pairs = []
    for gi, gt in enumerate(gt_boxes):
        for di, det in enumerate(det_boxes):
            iou_val = compute_iou(gt, det)
            if iou_val >= iou_thresh:
                pairs.append((iou_val, gi, di))
    pairs.sort(key=lambda x: x[0], reverse=True)

    best_matched_iou = 0.0
    for iou_val, gi, di in pairs:
        if gi in matched_gt or di in matched_det:
            continue
        matched_gt.add(gi)
        matched_det.add(di)
        best_matched_iou = max(best_matched_iou, iou_val)

    tp = len(matched_gt)
    fp = len(det_boxes) - len(matched_det)
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn, best_matched_iou


def aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df[["tp", "fp", "fn"]].isna().any().any():
        raise ValueError(
            "aggregate_metrics received rows with NaN TP/FP/FN; "
            "filter timing-only rows before calling."
        )
    grouped = df.groupby(["detector", "condition", "pass"])[
        ["tp", "fp", "fn", "inference_ms"]
    ].sum()
    counts = df.groupby(["detector", "condition", "pass"]).size()
    grouped["precision"] = grouped["tp"] / (grouped["tp"] + grouped["fp"]).clip(lower=1)
    grouped["recall"] = grouped["tp"] / (grouped["tp"] + grouped["fn"]).clip(lower=1)
    p = grouped["precision"]
    r = grouped["recall"]
    grouped["f1"] = 2 * p * r / (p + r).clip(lower=1e-9)
    grouped["mean_ms"] = grouped["inference_ms"] / counts
    return grouped.reset_index()
