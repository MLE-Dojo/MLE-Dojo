from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class HomeDataForMlCourseMetrics(CompetitionMetrics):
    """Metric class for Home Data for ML Course competition using RMSE of log-transformed SalePrice."""
    def __init__(self, value: str = "SalePrice", higher_is_better: bool = False):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Convert first column (Id) to string type
        y_true[y_true.columns[0]] = y_true[y_true.columns[0]].astype(str)
        y_pred[y_pred.columns[0]] = y_pred[y_pred.columns[0]].astype(str)

        # Sort by the first column (Id) and reset index
        y_true = y_true.sort_values(by=y_true.columns[0]).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=y_pred.columns[0]).reset_index(drop=True)

        # Calculate RMSE between the logarithm of the predicted and observed SalePrice values
        # Take natural log of the SalePrice column
        # y_true_log = np.log(y_true[self.value])
        # y_pred_log = np.log(y_pred[self.value])
        rmse = np.sqrt(np.mean((y_pred[self.value] - y_true[self.value]) ** 2))
        return rmse

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        
        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(
                f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows."
            )
        
        # Convert first column (Id) to string type - Do this early before checking columns
        try:
            submission[submission.columns[0]] = submission[submission.columns[0]].astype(str)
            ground_truth[ground_truth.columns[0]] = ground_truth[ground_truth.columns[0]].astype(str)
        except Exception as e:
             raise InvalidSubmissionError(f"Could not convert first column to string type. Error: {e}")
        
        # Check column existence (before accessing self.value)
        sub_cols = set(submission.columns)
        true_cols = set(ground_truth.columns) # Although ground truth columns are not strictly needed for submission validation, it's good practice if needed later.

        required_cols = {submission.columns[0], self.value} # Typically Id and the target variable
        if not required_cols.issubset(sub_cols):
             missing_cols = required_cols - sub_cols
             raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")

        # Check for NaN or Inf values in the prediction column
        pred_col = submission[self.value]
        if pred_col.isnull().any():
            raise InvalidSubmissionError(f"Submission column '{self.value}' contains missing (NaN) values.")
        if np.isinf(pred_col).any():
            raise InvalidSubmissionError(f"Submission column '{self.value}' contains infinite (inf) values.")

        # Check if all prediction values are greater than 0 (necessary for log)
        if (pred_col <= 0).any():
             raise InvalidSubmissionError(f"Submission column '{self.value}' contains non-positive values. Logarithm requires positive values.")

        # Sort the submission and ground truth by the first column (Id)
        try:
            submission = submission.sort_values(by=submission.columns[0]).reset_index(drop=True)
            ground_truth = ground_truth.sort_values(by=ground_truth.columns[0]).reset_index(drop=True)
        except Exception as e:
            raise InvalidSubmissionError(f"Could not sort dataframes by the first column. Error: {e}")

        # Check if the first column values are identical between submission and ground truth
        if (submission[submission.columns[0]].values != ground_truth[ground_truth.columns[0]].values).any():
            # Provide more specific feedback if possible (e.g., first mismatch)
            diff_mask = submission[submission.columns[0]].values != ground_truth[ground_truth.columns[0]].values
            first_diff_idx = np.where(diff_mask)[0]
            if len(first_diff_idx) > 0:
                 idx = first_diff_idx[0]
                 sub_id = submission[submission.columns[0]].iloc[idx]
                 gt_id = ground_truth[ground_truth.columns[0]].iloc[idx]
                 raise InvalidSubmissionError(f"First column values do not match between submission and ground truth after sorting. "
                                             f"First mismatch at index {idx}: submission ID='{sub_id}', ground truth ID='{gt_id}'. "
                                             f"Please ensure the first column values and order are identical.")
            else:
                 # Should not happen if .any() is true, but as a fallback
                 raise InvalidSubmissionError("First column values do not match between submission and ground truth after sorting. Please ensure the first column values are identical.")

        # Optional: Check for extra columns (depends on competition rules)
        # extra_cols = sub_cols - required_cols
        # if extra_cols:
        #     raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."