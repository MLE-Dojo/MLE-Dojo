from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class TabularPlaygroundSeriesJan2021Metrics(CompetitionMetrics):
    """Metric class for Tabular Playground Series Jan 2021 competition using RMSE"""

    def __init__(self, value: str = "target", higher_is_better: bool = False):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure the 'id' column is of string type for proper sorting
        y_true["id"] = y_true["id"].astype(str)
        y_pred["id"] = y_pred["id"].astype(str)
        # Sort both dataframes by the 'id' column
        y_true = y_true.sort_values(by="id").reset_index(drop=True)
        y_pred = y_pred.sort_values(by="id").reset_index(drop=True)
        # Calculate RMSE: sqrt(mean((y_true - y_pred)^2))
        diff = y_true[self.value] - y_pred[self.value]
        rmse = np.sqrt(np.mean(diff ** 2))
        return rmse

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Ensure the 'id' column is of string type for proper comparison
        submission["id"] = submission["id"].astype(str)
        ground_truth["id"] = ground_truth["id"].astype(str)

        # Sort the submission and ground truth by the 'id' column
        submission_sorted = submission.sort_values(by="id").reset_index(drop=True)
        ground_truth_sorted = ground_truth.sort_values(by="id").reset_index(drop=True)

        # Check if 'id' columns are identical
        if not (submission_sorted["id"].values == ground_truth_sorted["id"].values).all():
            raise InvalidSubmissionError("The 'id' column values do not match between submission and ground truth. Please ensure the 'id' values are identical and in the same order.")

        sub_cols = set(submission.columns)
        true_cols = set(ground_truth.columns)

        missing_cols = true_cols - sub_cols
        extra_cols = sub_cols - true_cols

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."