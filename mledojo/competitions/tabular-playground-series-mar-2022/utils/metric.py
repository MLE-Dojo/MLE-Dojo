from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class TabularPlaygroundSeriesMar2022Metrics(CompetitionMetrics):
    """Metric class for Tabular Playground Series Mar 2022 competition using Mean Absolute Error (MAE)."""
    def __init__(self, value: str = "congestion", higher_is_better: bool = False):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure the id column 'row_id' is of string type for correct sorting and matching
        y_true['row_id'] = y_true['row_id'].astype(str)
        y_pred['row_id'] = y_pred['row_id'].astype(str)
        # Sort both dataframes by the 'row_id' column
        y_true = y_true.sort_values(by='row_id').reset_index(drop=True)
        y_pred = y_pred.sort_values(by='row_id').reset_index(drop=True)
        # Calculate Mean Absolute Error (MAE)
        mae = np.mean(np.abs(y_true[self.value] - y_pred[self.value]))
        return mae

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Ensure 'row_id' column exists in both submission and ground truth
        if 'row_id' not in submission.columns:
            raise InvalidSubmissionError("Submission must contain the 'row_id' column.")
        if 'row_id' not in ground_truth.columns:
            raise InvalidSubmissionError("Ground truth must contain the 'row_id' column.")

        # Convert 'row_id' columns to string type for consistency
        submission['row_id'] = submission['row_id'].astype(str)
        ground_truth['row_id'] = ground_truth['row_id'].astype(str)
        # Sort the submission and ground truth by 'row_id'
        submission = submission.sort_values(by='row_id').reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by='row_id').reset_index(drop=True)

        # Check if 'row_id' values match between submission and ground truth
        if not (submission['row_id'].values == ground_truth['row_id'].values).all():
            raise InvalidSubmissionError("The 'row_id' values in submission do not match those in ground truth. Please ensure they are identical and in the same order.")

        # Check that required column for prediction exists
        if self.value not in submission.columns:
            raise InvalidSubmissionError(f"Submission is missing the required column: {self.value}.")
        if self.value not in ground_truth.columns:
            raise InvalidSubmissionError(f"Ground truth is missing the required column: {self.value}.")

        # Check for extra unexpected columns
        sub_cols = set(submission.columns)
        true_cols = set(ground_truth.columns)
        missing_cols = true_cols - sub_cols
        extra_cols = sub_cols - true_cols

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."