from typing import Any
import numpy as np
import pandas as pd
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

def compute_iou(box1, box2):
    # box format: [xmin, ymin, xmax, ymax]
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    if union_area == 0:
        return 0.0
    return intersection_area / union_area

def compute_average_precision(recalls, precisions):
    # Append sentinel values at the beginning and end
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    # Make precision monotonically decreasing
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    # Calculate area under precision-recall curve using numerical integration
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
    return ap

class VinbigdataChestXrayAbnormalitiesDetectionMetrics(CompetitionMetrics):
    """
    Metric class for VinBigData Chest Xray Abnormalities Detection competition using standard
    Pascal VOC 2010 mean Average Precision (mAP) at IoU > 0.4.
    """
    def __init__(self, value: str = "PredictionString", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Convert ID column (first column) to string type and sort both dataframes by ID
        id_col_true = y_true.columns[0]
        id_col_pred = y_pred.columns[0]
        y_true[id_col_true] = y_true[id_col_true].astype(str)
        y_pred[id_col_pred] = y_pred[id_col_pred].astype(str)
        y_true = y_true.sort_values(by=id_col_true).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=id_col_pred).reset_index(drop=True)
        
        # Dictionaries to hold ground truths and predictions per class across images.
        # Structure for ground truths: {class_id: list of dicts with keys: image_id, bbox, used}
        gt_by_class = {}
        # Structure for predictions: {class_id: list of dicts with keys: image_id, confidence, bbox}
        pred_by_class = {}
        
        num_images = y_true.shape[0]
        for index in range(num_images):
            image_id = y_true.iloc[index, 0]
            # Parse ground truth and predicted strings; they are space separated tokens.
            gt_str = y_true.iloc[index][self.value]
            pred_str = y_pred.iloc[index][self.value]
            
            # Function to parse a string into list of detections.
            # Each detection is a list: [class_id, confidence, xmin, ymin, xmax, ymax]
            def parse_detections(det_str, is_gt=False):
                dets = []
                if isinstance(det_str, str) and det_str.strip() != "":
                    tokens = det_str.strip().split()
                    # Each detection has 6 values
                    for i in range(0, len(tokens), 6):
                        try:
                            class_id = int(tokens[i])
                            confidence = float(tokens[i+1])
                            bbox = [float(tokens[i+2]), float(tokens[i+3]), float(tokens[i+4]), float(tokens[i+5])]
                            # For ground truth, ignore the confidence (set to 1.0)
                            if is_gt:
                                confidence = 1.0
                            dets.append((class_id, confidence, bbox))
                        except Exception:
                            continue
                return dets

            gt_detections = parse_detections(gt_str, is_gt=True)
            pred_detections = parse_detections(pred_str, is_gt=False)
            
            # Aggregate ground truth boxes per class
            for det in gt_detections:
                class_id, _, bbox = det
                if class_id not in gt_by_class:
                    gt_by_class[class_id] = []
                gt_by_class[class_id].append({
                    "image_id": image_id,
                    "bbox": bbox,
                    "used": False
                })
            # Aggregate predictions per class
            for det in pred_detections:
                class_id, confidence, bbox = det
                if class_id not in pred_by_class:
                    pred_by_class[class_id] = []
                pred_by_class[class_id].append({
                    "image_id": image_id,
                    "confidence": confidence,
                    "bbox": bbox
                })
        
        # Calculate Average Precision (AP) per class
        ap_list = []
        # Consider classes from 0 to 14 (15 classes), where class 14 (No finding) is also a valid detection.
        for cls in range(0, 15):
            gt_cls = gt_by_class.get(cls, [])
            pred_cls = pred_by_class.get(cls, [])
            
            npos = len(gt_cls)
            if npos == 0:
                # As per common practice, skip classes with no ground truths
                continue
            
            # Sort predictions by confidence descending
            pred_cls = sorted(pred_cls, key=lambda x: x["confidence"], reverse=True)
            tp = np.zeros(len(pred_cls))
            fp = np.zeros(len(pred_cls))
            
            # Create a lookup for ground truths per image for this class
            gt_per_image = {}
            for item in gt_cls:
                img = item["image_id"]
                if img not in gt_per_image:
                    gt_per_image[img] = []
                gt_per_image[img].append(item)
            
            # For each prediction, mark true positive or false positive
            for i, pred in enumerate(pred_cls):
                image_id = pred["image_id"]
                pred_box = pred["bbox"]
                # Get ground truths for this image and class
                gts = gt_per_image.get(image_id, [])
                best_iou = 0.0
                best_gt = None
                for gt in gts:
                    if gt["used"]:
                        continue
                    iou = compute_iou(pred_box, gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gt
                if best_iou >= 0.4 and best_gt is not None:
                    tp[i] = 1
                    best_gt["used"] = True
                else:
                    fp[i] = 1

            # Compute cumulative true positives and false positives
            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            recalls = cum_tp / float(npos)
            precisions = cum_tp / (cum_tp + cum_fp + 1e-6)  # avoid division by zero
            ap = compute_average_precision(recalls, precisions)
            ap_list.append(ap)
        
        # If no classes had ground truths, return 0.0
        mAP = np.mean(ap_list) if len(ap_list) > 0 else 0.0
        return mAP

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        id_col_sub = submission.columns[0]
        id_col_gt = ground_truth.columns[0]

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(
                f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). "
                "Please ensure both have the same number of rows."
            )

        submission[id_col_sub] = submission[id_col_sub].astype(str)
        ground_truth[id_col_gt] = ground_truth[id_col_gt].astype(str)

        submission = submission.sort_values(by=id_col_sub).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=id_col_gt).reset_index(drop=True)

        if (submission[id_col_sub].values != ground_truth[id_col_gt].values).any():
            raise InvalidSubmissionError(
                "ID column values do not match between submission and ground truth. Please ensure the IDs are identical and in the same order."
            )

        sub_cols = set(submission.columns)
        true_cols = set(ground_truth.columns)

        missing_cols = true_cols - sub_cols
        extra_cols = sub_cols - true_cols

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."