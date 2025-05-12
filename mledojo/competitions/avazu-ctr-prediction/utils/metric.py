from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError
from sklearn.metrics import log_loss

class AvazuCtrPredictionMetrics(CompetitionMetrics):
    """Metric class for Avazu CTR Prediction competition using Logarithmic Loss"""
    def __init__(self, value: str = "click", higher_is_better: bool = False):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Convert first column to string type
        y_true[y_true.columns[0]] = y_true[y_true.columns[0]].astype(str)
        y_pred[y_pred.columns[0]] = y_pred[y_pred.columns[0]].astype(str)
        
        # Sort by the id column (first column)
        y_true = y_true.sort_values(by=y_true.columns[0]).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=y_pred.columns[0]).reset_index(drop=True)
        
        # Compute Logarithmic Loss
        # Ensure that predictions are within (0,1) bounds to avoid errors with log(0)
        eps = 1e-15
        preds = np.clip(y_pred[self.value].values, eps, 1 - eps)
        return log_loss(y_true[self.value].values, preds)

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        id_col_name = submission.columns[0]
        gt_id_col_name = ground_truth.columns[0]

        # Ensure first column names are the same before proceeding
        if id_col_name != gt_id_col_name:
             raise InvalidSubmissionError(f"Submission ID column name ('{id_col_name}') does not match ground truth ID column name ('{gt_id_col_name}').")

        # Convert first column to string type for consistent sorting
        try:
            submission[id_col_name] = submission[id_col_name].astype(str)
            ground_truth[gt_id_col_name] = ground_truth[gt_id_col_name].astype(str)
        except Exception as e:
             raise InvalidSubmissionError(f"Failed to convert ID columns to string type. Error: {e}")

        # Sort the submission and ground truth by the string representation of the first column
        submission_sorted = submission.sort_values(by=id_col_name).reset_index(drop=True)
        ground_truth_sorted = ground_truth.sort_values(by=gt_id_col_name).reset_index(drop=True)

        # Attempt to convert sorted ID columns to numeric for robust comparison
        try:
            sub_ids_numeric = submission_sorted[id_col_name].astype(np.float64)
            gt_ids_numeric = ground_truth_sorted[gt_id_col_name].astype(np.float64)
        except ValueError as e:
            raise InvalidSubmissionError(f"Could not convert ID column ('{id_col_name}') to numeric type after sorting. Please ensure all IDs are valid numbers. Error: {e}")
        except Exception as e:
            raise InvalidSubmissionError(f"An unexpected error occurred during numeric conversion of ID columns. Error: {e}")


        # Check if numeric first columns are approximately equal (handles float differences)
        if not np.allclose(sub_ids_numeric.values, gt_ids_numeric.values):
            # Find the first mismatch for better error reporting
            diff_indices = np.where(~np.isclose(sub_ids_numeric.values, gt_ids_numeric.values))[0]
            first_diff_idx = diff_indices[0] if len(diff_indices) > 0 else 0 # Fallback to 0 if somehow isclose fails but allclose doesn't

            error_msg = (
                f"ID column ('{id_col_name}') values do not match between submission and ground truth after sorting, even with numeric tolerance. "
                f"First mismatch found near index {first_diff_idx}: "
                f"Submission ID (as string): '{submission_sorted.iloc[first_diff_idx, 0]}' "
                f"vs Ground Truth ID (as string): '{ground_truth_sorted.iloc[first_diff_idx, 0]}'. "
                f"Please ensure the ID columns contain the same set of identifiers in the same order."
            )
            raise InvalidSubmissionError(error_msg)

        # --- Column Name Validation ---
        sub_cols = set(submission_sorted.columns)
        true_cols = set(ground_truth_sorted.columns)
        # Check if the target column (value) exists in the submission
        if self.value not in sub_cols:
             raise InvalidSubmissionError(f"Required target column '{self.value}' not found in submission.")
        
        # Allow only ID and target columns, raise error for any others
        expected_cols = {id_col_name, self.value}
        extra_cols = sub_cols - expected_cols
        if extra_cols:
            raise InvalidSubmissionError(f"Unexpected columns found in submission: {', '.join(extra_cols)}. Only columns '{id_col_name}' and '{self.value}' are expected.")
        
        # Check for missing target column (already covered by 'in' check, but good practice)
        missing_cols = expected_cols - sub_cols 
        if missing_cols: # Should not happen if self.value check passed, but for completeness
             raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")


        return "Submission is valid."