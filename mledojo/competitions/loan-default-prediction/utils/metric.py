from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class LoanDefaultPredictionMetrics(CompetitionMetrics):
    """Metric class for loan-default-prediction competition using Mean Absolute Error (MAE)"""
    def __init__(self, value: str = "loss", higher_is_better: bool = False):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Convert id column to string and sort by it
        id_col_true = y_true.columns[0]
        id_col_pred = y_pred.columns[0]
        y_true[id_col_true] = y_true[id_col_true].astype(str)
        y_pred[id_col_pred] = y_pred[id_col_pred].astype(str)
        y_true = y_true.sort_values(by=id_col_true).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=id_col_pred).reset_index(drop=True)
        # Compute MAE using the 'loss' column (specified by self.value)
        mae = np.mean(np.abs(y_true[self.value] - y_pred[self.value]))
        return mae

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Convert first column (id column) to string type for both submission and ground truth
        submission_id = submission.columns[0]
        ground_truth_id = ground_truth.columns[0]
        submission[submission_id] = submission[submission_id].astype(str)
        ground_truth[ground_truth_id] = ground_truth[ground_truth_id].astype(str)
        # Sort submission and ground truth by the id column
        submission = submission.sort_values(by=submission_id).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=ground_truth_id).reset_index(drop=True)

        # Check if the id columns are identical
        if not np.array_equal(submission[submission_id].values, ground_truth[ground_truth_id].values):
            raise InvalidSubmissionError("The 'id' column values do not match between submission and ground truth. Please ensure the first column values are identical.")

        required_columns = {ground_truth_id, "loss"}
        submission_columns = set(submission.columns)
        ground_truth_columns = set(ground_truth.columns)

        missing_cols = required_columns - submission_columns
        extra_cols = submission_columns - required_columns
            
        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."