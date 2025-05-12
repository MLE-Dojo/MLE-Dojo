from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class SantanderValuePredictionChallengeMetrics(CompetitionMetrics):
    """Metric class for Santander Value Prediction Challenge using Root Mean Squared Logarithmic Error (RMSLE)"""
    def __init__(self, value: str = "target", higher_is_better: bool = False):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Convert ID columns to string type
        y_true['ID'] = y_true['ID'].astype(str)
        y_pred['ID'] = y_pred['ID'].astype(str)
        # Sort values by the ID column
        y_true = y_true.sort_values(by='ID').reset_index(drop=True)
        y_pred = y_pred.sort_values(by='ID').reset_index(drop=True)
        
        # Calculate RMSLE
        # Add 1 to avoid log(0)
        log_true = np.log(y_true[self.value] + 1)
        log_pred = np.log(y_pred[self.value] + 1)
        rmsle = np.sqrt(np.mean((log_pred - log_true) ** 2))
        return rmsle

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Convert 'ID' column to string type
        submission['ID'] = submission['ID'].astype(str)
        ground_truth['ID'] = ground_truth['ID'].astype(str)
        # Sort both submission and ground truth by 'ID'
        submission_sorted = submission.sort_values(by='ID').reset_index(drop=True)
        ground_truth_sorted = ground_truth.sort_values(by='ID').reset_index(drop=True)

        # Check if 'ID' columns are identical
        if not (submission_sorted['ID'].values == ground_truth_sorted['ID'].values).all():
            raise InvalidSubmissionError("ID values do not match between submission and ground truth. Please ensure the ID column values are identical and in the same order.")

        # Check for missing values in submission
        if submission.isnull().any().any():
            raise InvalidSubmissionError("Submission contains missing values. Please ensure all values are filled.")

        sub_cols = set(submission.columns)
        true_cols = set(ground_truth.columns)

        missing_cols = true_cols - sub_cols
        extra_cols = sub_cols - true_cols

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."