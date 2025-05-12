from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class KuzushijiRecognitionMetrics(CompetitionMetrics):
    """
    Metric class for Kuzushiji Recognition competition.
    
    The evaluation uses a modified F1 Score.
    A true positive is counted when a predicted label with its center point (X, Y) lies within 
    a ground truth bounding box and the label matches.
    
    Ground truth format (per image): a string of space separated tokens in the order:
      label X Y Width Height  (repeated for each object)
    Submission format (per image): a string of space separated tokens in the order:
      label X Y  (repeated for each predicted object)
    """
    def __init__(self, value: str = "labels", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure first column (image_id) is treated as string and properly sorted.
        # Convert potential non-string values to empty strings for safe processing.
        y_true[y_true.columns[0]] = y_true[y_true.columns[0]].astype(str)
        y_pred[y_pred.columns[0]] = y_pred[y_pred.columns[0]].astype(str)
        y_true = y_true.sort_values(by=y_true.columns[0]).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=y_pred.columns[0]).reset_index(drop=True)

        # Handle potential NaN/None in label columns by replacing with empty string
        label_col_index = y_true.columns.get_loc(self.value) # Assuming self.value is the name of the label column
        y_true.iloc[:, label_col_index] = y_true.iloc[:, label_col_index].fillna('').astype(str)
        y_pred.iloc[:, label_col_index] = y_pred.iloc[:, label_col_index].fillna('').astype(str)

        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        n = len(y_true)
        for i in range(n):
            # Get image id, ground truth labels, and predicted labels
            image_id_true = y_true.iloc[i, 0]
            image_id_pred = y_pred.iloc[i, 0]
            # Ensure matching image_ids per row (already checked in validate_submission, but good for safety)
            if image_id_true != image_id_pred:
                # This case should ideally be caught by validate_submission
                raise InvalidSubmissionError(f"Image IDs do not match during evaluation for row {i}: {image_id_true} vs {image_id_pred}.")
            
            gt_str = y_true.iloc[i][self.value]
            pred_str = y_pred.iloc[i][self.value]
            
            # Parse ground truth boxes.
            gt_boxes = []
            # Use gt_str.strip() to handle strings with only whitespace
            if gt_str.strip(): 
                gt_tokens = gt_str.strip().split()
                if len(gt_tokens) % 5 != 0:
                    # This should ideally be caught by ground truth validation if possible, 
                    # but handling here makes evaluation robust.
                    raise ValueError(f"Ground truth format error (expected label X Y W H components) for image {image_id_true}.")
                # each object: [label, X, Y, Width, Height]
                for j in range(0, len(gt_tokens), 5):
                    label = gt_tokens[j]
                    try:
                        x = float(gt_tokens[j+1])
                        y = float(gt_tokens[j+2])
                        w = float(gt_tokens[j+3])
                        h = float(gt_tokens[j+4])
                        if w < 0 or h < 0: # Basic sanity check for width/height
                             raise ValueError("Width and Height must be non-negative.")
                    except ValueError as e:
                        raise ValueError(f"Numeric conversion error or invalid value in ground truth for image {image_id_true}: {e}")
                    gt_boxes.append({
                        "label": label,
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "matched": False
                    })

            # Parse predicted points.
            pred_points = []
            # Use pred_str.strip() to handle strings with only whitespace
            if pred_str.strip():
                pred_tokens = pred_str.strip().split()
                # Format check (already done in validate_submission, but kept for robustness)
                if len(pred_tokens) % 3 != 0:
                    raise InvalidSubmissionError(f"Submission format error (expected label X Y components) for image {image_id_pred}.")
                # Max prediction check (already done in validate_submission)
                num_predictions = len(pred_tokens) // 3
                if num_predictions > 1200:
                     raise InvalidSubmissionError(f"Exceeded maximum prediction limit (1200) for image {image_id_pred}: found {num_predictions}.")

                # each prediction: [label, X, Y]
                for j in range(0, len(pred_tokens), 3):
                    label = pred_tokens[j]
                    try:
                        x = float(pred_tokens[j+1])
                        y = float(pred_tokens[j+2])
                    except ValueError:
                         # This should be caught by validate_submission
                        raise InvalidSubmissionError(f"Numeric conversion error in submission for image {image_id_pred}.")
                    pred_points.append({
                        "label": label,
                        "x": x,
                        "y": y,
                        "matched": False # Note: 'matched' here isn't strictly needed for preds but mirrors gt structure
                    })

            tp = 0
            fp_count_for_image = 0 # Counter for false positives for this specific image
            # For each predicted point, attempt to match *one* unused ground truth box.
            for pred in pred_points:
                matched_to_gt = False
                for gt_idx, gt in enumerate(gt_boxes):
                    # Check if ground truth box is already matched, if label matches, and if point is inside box
                    if not gt["matched"] and \
                       pred["label"] == gt["label"] and \
                       (gt["x"] <= pred["x"] < gt["x"] + gt["w"]) and \
                       (gt["y"] <= pred["y"] < gt["y"] + gt["h"]):
                        
                        tp += 1
                        gt["matched"] = True  # Mark this GT box as used
                        matched_to_gt = True
                        break # Stop searching for matches for this prediction once found
                
                if not matched_to_gt:
                    fp_count_for_image += 1 # This prediction did not match any available GT box

            # False negatives are the ground truth boxes that were never matched.
            fn = len([gt for gt in gt_boxes if not gt["matched"]])
            
            total_tp += tp
            total_fp += fp_count_for_image # Add the FPs found for this image
            total_fn += fn

        # If no predictions and no ground truths across all images, consider perfect score.
        if total_tp == 0 and total_fp == 0 and total_fn == 0:
             # Check if y_true and y_pred actually contained only empty strings after stripping
            all_gt_empty = all(s.strip() == '' for s in y_true[self.value])
            all_pred_empty = all(s.strip() == '' for s in y_pred[self.value])
            if all_gt_empty and all_pred_empty:
                 return 1.0
            # Otherwise, if there were non-empty but invalid entries leading to 0 counts, F1 is 0
            else:
                 return 0.0

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        return f1

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            # This check might not be strictly necessary if validation only uses submission, 
            # but good practice if GT info (like image IDs) is needed.
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}).")

        # Check column presence before accessing them
        if submission.columns[0] not in submission:
             raise InvalidSubmissionError(f"Submission missing expected first column (image_id).")
        if self.value not in submission.columns:
             raise InvalidSubmissionError(f"Submission missing expected label column: '{self.value}'.")
        if ground_truth.columns[0] not in ground_truth:
             raise InvalidSubmissionError(f"Ground truth missing expected first column (image_id).")
         # We don't strictly need the label column from ground_truth for validation itself


        # Convert first column (assumed image_id) to string type for both.
        try:
            submission[submission.columns[0]] = submission[submission.columns[0]].astype(str)
            ground_truth[ground_truth.columns[0]] = ground_truth[ground_truth.columns[0]].astype(str)
        except Exception as e:
             raise InvalidSubmissionError(f"Failed to convert image_id column to string: {e}")

        sub_copy = submission.sort_values(by=submission.columns[0]).reset_index(drop=True)
        gt_copy = ground_truth.sort_values(by=ground_truth.columns[0]).reset_index(drop=True)

        # Check if image_id values are identical and in the same order after sorting
        if not sub_copy[sub_copy.columns[0]].equals(gt_copy[gt_copy.columns[0]]):
            # Provide more specific feedback if possible (e.g., missing/extra IDs)
            sub_ids = set(sub_copy[sub_copy.columns[0]])
            gt_ids = set(gt_copy[gt_copy.columns[0]])
            missing_in_sub = gt_ids - sub_ids
            extra_in_sub = sub_ids - gt_ids
            error_msg = "Image IDs do not match between submission and ground truth."
            if missing_in_sub:
                error_msg += f" Missing IDs in submission: {list(missing_in_sub)[:5]}..." # Show a few examples
            if extra_in_sub:
                 error_msg += f" Extra IDs in submission: {list(extra_in_sub)[:5]}..."
            if len(sub_ids) != len(gt_ids):
                 error_msg += f" Different number of unique IDs ({len(sub_ids)} vs {len(gt_ids)})."

            raise InvalidSubmissionError(error_msg + " Please ensure the first column values correspond exactly.")

        # Check column names (if strict matching is required)
        # If only image_id and self.value are required, check for those specifically.
        required_cols = {submission.columns[0], self.value}
        sub_cols = set(submission.columns)
        if not required_cols.issubset(sub_cols):
             missing = required_cols - sub_cols
             raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing)}.")
        # Optionally check for extra columns if the format must be exact
        # extra_cols = sub_cols - required_cols
        # if extra_cols:
        #     raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")


        # Validate format and content of the prediction string column (self.value)
        label_col = self.value
        for idx, row in sub_copy.iterrows():
            image_id = row[sub_copy.columns[0]]
            pred_str = row[label_col]

            # Handle potential non-string types (like NaN if read from CSV)
            if pd.isna(pred_str):
                 pred_str = "" # Treat NaN/None as empty string
            elif not isinstance(pred_str, str):
                 raise InvalidSubmissionError(f"Invalid data type in submission column '{label_col}' for image {image_id}. Expected string, got {type(pred_str)}.")

            pred_str = pred_str.strip()
            if not pred_str: # Empty string is valid (no predictions)
                continue

            pred_tokens = pred_str.split()

            # Check 1: Format must be triplets (label X Y)
            if len(pred_tokens) % 3 != 0:
                raise InvalidSubmissionError(f"Submission format error for image {image_id}. Expected groups of 'label X Y', but found {len(pred_tokens)} items. Problem segment: '{' '.join(pred_tokens[:5])}...'.")

            # Check 2: Max predictions per image
            num_predictions = len(pred_tokens) // 3
            if num_predictions > 1200:
                raise InvalidSubmissionError(f"Exceeded maximum prediction limit (1200) for image {image_id}: found {num_predictions}.")

            # Check 3: Numeric coordinates
            for j in range(1, len(pred_tokens), 3): # Check X coordinates
                try:
                    float(pred_tokens[j])
                except ValueError:
                    raise InvalidSubmissionError(f"Non-numeric X coordinate '{pred_tokens[j]}' found for image {image_id} in item {j//3 + 1} ('{pred_tokens[j-1]} {pred_tokens[j]} {pred_tokens[j+1]}').")
            for j in range(2, len(pred_tokens), 3): # Check Y coordinates
                 try:
                     float(pred_tokens[j])
                 except ValueError:
                     raise InvalidSubmissionError(f"Non-numeric Y coordinate '{pred_tokens[j]}' found for image {image_id} in item {j//3 + 1} ('{pred_tokens[j-1]} {pred_tokens[j-1]} {pred_tokens[j]}').")


        return "Submission is valid."