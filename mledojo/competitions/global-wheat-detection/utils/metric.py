from typing import Any, List, Tuple
import numpy as np
import pandas as pd
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

def parse_pred_boxes(s: str) -> List[Tuple[float, List[float]]]:
    """
    Parse the prediction string for predicted boxes.
    Expected format: "conf x y width height [conf x y width height ...]".
    Returns a list of tuples (confidence, [x, y, width, height]).
    Handles potentially malformed strings by ignoring trailing tokens that don't form a complete box.
    """
    if pd.isna(s) or s.strip() == "":
        return []
    tokens = list(map(float, s.split()))
    boxes = []
    # Ensure we only process full 5-token sets
    num_complete_boxes = len(tokens) // 5
    for i in range(0, num_complete_boxes * 5, 5):
        conf = tokens[i]
        box = tokens[i+1:i+5]
        # Basic validation: ensure box has 4 coordinates
        if len(box) == 4:
            boxes.append((conf, box))
        # Optional: Add logging here if you want to know about ignored partial boxes
    return boxes

def parse_gt_boxes(s: str) -> List[List[float]]:
    """
    Parse the ground truth string for bounding boxes.
    Expected format: "1.0 x y width height [1.0 x y width height ...]".
    Returns a list of boxes, each as [x, y, width, height].
    Handles potentially malformed strings by ignoring trailing tokens that don't form a complete box.
    """
    if pd.isna(s) or s.strip() == "":
        return []
    tokens = list(map(float, s.split()))
    boxes = []
    # Ensure we only process full 5-token sets (conf x y w h)
    num_complete_boxes = len(tokens) // 5
    for i in range(0, num_complete_boxes * 5, 5):
        # The first token (index i) is confidence (assumed 1.0 for GT), we skip it.
        box = tokens[i+1:i+5]
        # Basic validation: ensure box has 4 coordinates
        if len(box) == 4:
            boxes.append(box)
        # Optional: Add logging here if you want to know about ignored partial boxes
    return boxes

def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.
    Each box is defined as [x, y, width, height].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    
    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    interArea = inter_width * inter_height
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    unionArea = boxAArea + boxBArea - interArea
    if unionArea == 0:
        return 0.0
    return interArea / unionArea

def compute_image_ap(gt_str: str, pred_str: str, thresholds: List[float]) -> float:
    """
    Compute the average precision for a single image.
    """
    # Use the updated, more robust parsing functions
    gt_boxes = parse_gt_boxes(gt_str)
    
    # Important: if there are no ground truth boxes, image AP is 0 regardless of predictions.
    # This also handles cases where gt_str was empty or malformed resulting in empty gt_boxes.
    if len(gt_boxes) == 0:
        return 0.0

    # Use the updated, more robust parsing function
    pred_boxes = parse_pred_boxes(pred_str)
    
    # Sort predicted boxes by descending confidence. Handles empty pred_boxes gracefully.
    pred_boxes = sorted(pred_boxes, key=lambda x: x[0], reverse=True)
    
    ap_scores = []
    num_preds = len(pred_boxes)
    num_gts = len(gt_boxes)

    for thr in thresholds:
        # Reset matching status for each threshold
        matched_gt = [False] * num_gts
        tp = 0
        
        # Keep track of which predictions were matched to avoid double counting FPs later
        # This isn't strictly needed for the current TP/FP/FN calculation method, 
        # but can be useful for debugging or alternative precision metrics.
        # matched_pred = [False] * num_preds 

        # Match predictions to ground truth boxes
        for pred_idx, (conf, p_box) in enumerate(pred_boxes):
            best_iou = 0.0
            best_gt_idx = -1
            # Find the best matching unmatched GT box above the threshold
            for gt_idx, gt_box in enumerate(gt_boxes):
                if not matched_gt[gt_idx]:
                    iou_val = compute_iou(p_box, gt_box)
                    # Check if IoU meets threshold and is the best match so far for this prediction
                    if iou_val >= thr and iou_val > best_iou:
                        best_iou = iou_val
                        best_gt_idx = gt_idx
            
            # If a match was found, mark GT as matched and increment TP
            if best_gt_idx >= 0:
                tp += 1
                matched_gt[best_gt_idx] = True
                # matched_pred[pred_idx] = True # Mark prediction as matched (TP)

        # Calculate FP and FN based on the number of matches (tp)
        fp = num_preds - tp  # Predictions that didn't match any GT
        fn = num_gts - tp    # GTs that weren't matched by any prediction
        
        # Calculate precision for this threshold according to the formula TP / (TP + FP + FN)
        denom = tp + fp + fn
        precision = tp / denom if denom > 0 else 0.0
        ap_scores.append(precision)
    
    # Return the mean precision across all thresholds
    return float(np.mean(ap_scores)) if ap_scores else 0.0

class GlobalWheatDetectionMetrics(CompetitionMetrics):
    """
    Metric class for Global Wheat Detection competition.
    Evaluation is based on mean average precision (mAP) over IoU thresholds from 0.5 to 0.75.
    """
    def __init__(self, value: str = "PredictionString", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value  # Column used to calculate the score

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure the identifier column (first column: image_id) is of type string.
        y_true[y_true.columns[0]] = y_true[y_true.columns[0]].astype(str)
        y_pred[y_pred.columns[0]] = y_pred[y_pred.columns[0]].astype(str)
        
        # Sort both DataFrames by the image_id column.
        y_true = y_true.sort_values(by=y_true.columns[0]).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=y_pred.columns[0]).reset_index(drop=True)
        
        # Merge on image_id (first column)
        merged = pd.merge(y_true, y_pred, on=y_true.columns[0], suffixes=('_gt', '_pred'))
        
        # Define IoU thresholds from 0.5 to 0.75 (step size 0.05)
        thresholds = [round(x, 2) for x in np.arange(0.5, 0.76, 0.05)]
        
        image_scores = []
        # Assuming ground truth bounding boxes are in the column with the same name as self.value in y_true,
        # and predicted bounding boxes are in the column self.value in y_pred.
        for _, row in merged.iterrows():
            gt_str = row[self.value + "_gt"]
            pred_str = row[self.value + "_pred"]
            image_ap = compute_image_ap(gt_str, pred_str, thresholds)
            image_scores.append(image_ap)
        
        # Return mean AP over all images.
        return float(np.mean(image_scores))
    
    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        # Check that required columns exist
        required_columns = {"image_id", self.value}
        submission_cols = set(submission.columns)
        ground_truth_cols = set(ground_truth.columns)
        
        missing_in_submission = required_columns - submission_cols
        if missing_in_submission:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_in_submission)}.")
        missing_in_truth = required_columns - ground_truth_cols
        if missing_in_truth:
            raise InvalidSubmissionError(f"Missing required columns in ground truth: {', '.join(missing_in_truth)}.")
        
        # Convert identifier (first column: image_id) to string for both DataFrames.
        submission["image_id"] = submission["image_id"].astype(str)
        ground_truth["image_id"] = ground_truth["image_id"].astype(str)
        
        # Sort the DataFrames by the "image_id" column.
        submission_sorted = submission.sort_values(by="image_id").reset_index(drop=True)
        ground_truth_sorted = ground_truth.sort_values(by="image_id").reset_index(drop=True)
        
        # Verify that the image_id values are identical and in the same order.
        if not np.array_equal(submission_sorted["image_id"].values, ground_truth_sorted["image_id"].values):
            raise InvalidSubmissionError("The image_id values in submission and ground truth do not match. Please ensure they are identical and in the same order.")
        
        return "Submission is valid."