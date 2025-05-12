from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class PubgFinishPlacementPredictionMetrics(CompetitionMetrics):
    """Metric class for PUBG Finish Placement Prediction competition using Mean Absolute Error"""
    def __init__(self, value: str = "winPlacePerc", higher_is_better: bool = False):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Convert the first column (assumed to be the Id column) to string type
        id_col = y_true.columns[0]
        y_true[id_col] = y_true[id_col].astype(str)
        y_pred[id_col] = y_pred[id_col].astype(str)

        # Sort the ground truth and submission dataframes by the Id column
        y_true = y_true.sort_values(by=id_col).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=id_col).reset_index(drop=True)

        # Calculate Mean Absolute Error between predicted and true values of winPlacePerc
        mae = np.mean(np.abs(y_true[self.value] - y_pred[self.value]))
        return float(mae)

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Convert the first column (Id) to string type for both dataframes
        submission[submission.columns[0]] = submission[submission.columns[0]].astype(str)
        ground_truth[ground_truth.columns[0]] = ground_truth[ground_truth.columns[0]].astype(str)

        # Sort the submission and ground truth by the first column
        submission = submission.sort_values(by=submission.columns[0])
        ground_truth = ground_truth.sort_values(by=ground_truth.columns[0])

        # Check if first column values match between submission and ground truth
        if (submission[submission.columns[0]].values != ground_truth[ground_truth.columns[0]].values).any():
            raise InvalidSubmissionError("First column values do not match between submission and ground truth. Please ensure the first column values are identical.")

        sub_cols = set(submission.columns)
        true_cols = set(ground_truth.columns)

        missing_cols = true_cols - sub_cols
        extra_cols = sub_cols - true_cols

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."