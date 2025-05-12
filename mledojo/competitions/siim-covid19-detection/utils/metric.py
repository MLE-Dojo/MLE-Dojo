from typing import Any, List, Tuple, Dict
import numpy as np
import pandas as pd
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

def parse_predictions(pred_str: str) -> List[Tuple[str, float, List[float]]]:
    """
    Parse the prediction string into list of (class, confidence, bbox) tuples.
    The bbox is a list of 4 floats: [xmin, ymin, xmax, ymax].
    If pred_str is empty or NaN, returns an empty list.
    """
    if pd.isna(pred_str) or pred_str.strip() == "":
        return []
    tokens = pred_str.strip().split()
    predictions = []
    # Expect groups of 6 tokens: class, confidence, xmin, ymin, xmax, ymax
    if len(tokens) % 6 != 0:
        # If the format is not as expected, return empty - validation should have caught this.
        return predictions
    for i in range(0, len(tokens), 6):
        cls = tokens[i]
        try:
            conf = float(tokens[i+1])
            bbox = [float(tokens[i+2]), float(tokens[i+3]), float(tokens[i+4]), float(tokens[i+5])]
        except ValueError:
            # In case conversion fails, skip this detection
            continue
        predictions.append((cls, conf, bbox))
    return predictions

def iou(boxA: List[float], boxB: List[float]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    Each box is defined as [xmin, ymin, xmax, ymax].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea
    if unionArea <= 0:
        return 0.0

    return interArea / unionArea

def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Compute average precision (AP) given recall and precision curves.
    This uses the 101-point interpolation as in VOC challenge.
    """
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    
    # Make precision monotonically decreasing
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    # Find points where recall changes
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    
    ap = 0.0
    for i in indices:
        ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]
    return ap

class SiimCovid19DetectionMetrics(CompetitionMetrics):
    """
    Metric class for SIIM-COVID19-Detection competition.
    Evaluation is based on the standard PASCAL VOC 2010 mean Average Precision (mAP) at IoU > 0.5.
    """
    def __init__(self, value: str = "PredictionString", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure the id columns are string type
        y_true[y_true.columns[0]] = y_true[y_true.columns[0]].astype(str)
        y_pred[y_pred.columns[0]] = y_pred[y_pred.columns[0]].astype(str)

        # Sort both dataframes by the identifier column (assumed to be the first column)
        id_col = y_true.columns[0]
        y_true = y_true.sort_values(by=id_col).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=id_col).reset_index(drop=True)

        # Build ground truth and prediction dictionaries by class.
        # Each dictionary maps a class label to a list of tuples.
        # For ground truth: (image_id, bbox, detected_flag)
        # For predictions: (image_id, confidence, bbox)
        gt_by_class: Dict[str, List[Tuple[str, List[float], bool]]] = {}
        pred_by_class: Dict[str, List[Tuple[str, float, List[float]]]] = {}

        # Process ground truth
        for idx, row in y_true.iterrows():
            image_id = row[id_col]
            pred_str = row[self.value]
            gt_list = parse_predictions(pred_str)
            for det in gt_list:
                cls, _, bbox = det
                if cls not in gt_by_class:
                    gt_by_class[cls] = []
                gt_by_class[cls].append((image_id, bbox, False))
        
        # Process predictions
        for idx, row in y_pred.iterrows():
            image_id = row[id_col]
            pred_str = row[self.value]
            pred_list = parse_predictions(pred_str)
            for det in pred_list:
                cls, conf, bbox = det
                if cls not in pred_by_class:
                    pred_by_class[cls] = []
                pred_by_class[cls].append((image_id, conf, bbox))
        
        # Calculate AP for each class
        ap_per_class = []
        all_classes = set(list(gt_by_class.keys()) + list(pred_by_class.keys()))
        for cls in all_classes:
            gt_for_cls = gt_by_class.get(cls, [])
            pred_for_cls = pred_by_class.get(cls, [])
            # Count total ground truth boxes for this class
            n_gt = len(gt_for_cls)
            if n_gt == 0:
                # If there are no ground truth boxes for this class, skip AP calculation
                continue
            # Sort predictions by descending confidence
            pred_for_cls = sorted(pred_for_cls, key=lambda x: x[1], reverse=True)
            tp = np.zeros(len(pred_for_cls))
            fp = np.zeros(len(pred_for_cls))
            # For bookkeeping, create a dict to hold gt detections for each image for this class
            gt_by_image = {}
            for image_id, bbox, detected in gt_for_cls:
                if image_id not in gt_by_image:
                    gt_by_image[image_id] = []
                gt_by_image[image_id].append({'bbox': bbox, 'detected': False})
            
            # Process each prediction
            for idx_pred, (image_id, conf, bbox_pred) in enumerate(pred_for_cls):
                gt_detections = gt_by_image.get(image_id, [])
                iou_max = 0.0
                jmax = -1
                for j, gt_det in enumerate(gt_detections):
                    iou_score = iou(bbox_pred, gt_det['bbox'])
                    if iou_score > iou_max:
                        iou_max = iou_score
                        jmax = j
                if iou_max >= 0.5:
                    if not gt_detections[jmax]['detected']:
                        tp[idx_pred] = 1  # True positive
                        gt_detections[jmax]['detected'] = True
                    else:
                        fp[idx_pred] = 1  # Duplicate detection (false positive)
                else:
                    fp[idx_pred] = 1  # IoU too low, false positive
            
            # Compute precision and recall
            cumul_tp = np.cumsum(tp)
            cumul_fp = np.cumsum(fp)
            recalls = cumul_tp / n_gt
            precisions = np.divide(cumul_tp, (cumul_tp + cumul_fp + 1e-6))
            ap = compute_ap(recalls, precisions)
            ap_per_class.append(ap)
        
        # If no classes had ground truth, return 0
        if len(ap_per_class) == 0:
            return 0.0
        # Mean Average Precision over classes
        mAP = np.mean(ap_per_class)
        return mAP

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Convert the identifier column (first column) to string type
        id_col = submission.columns[0]
        submission[id_col] = submission[id_col].astype(str)
        ground_truth[ground_truth.columns[0]] = ground_truth[ground_truth.columns[0]].astype(str)
        # Sort the submission and ground truth by their identifier column
        submission = submission.sort_values(by=id_col).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=ground_truth.columns[0]).reset_index(drop=True)

        # Check if identifier columns are identical
        if not np.array_equal(submission[id_col].values, ground_truth[ground_truth.columns[0]].values):
            raise InvalidSubmissionError("Identifier column values do not match between submission and ground truth. Please ensure the first column values are identical.")

        sub_cols = set(submission.columns)
        true_cols = set(ground_truth.columns)

        missing_cols = true_cols - sub_cols
        extra_cols = sub_cols - true_cols

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."