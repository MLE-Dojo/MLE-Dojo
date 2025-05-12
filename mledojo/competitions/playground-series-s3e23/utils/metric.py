from typing import Any
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class PlaygroundSeriesS3e23Metrics(CompetitionMetrics):
    """Metric class for playground-series-s3e23 competition using Area Under the ROC Curve."""
    def __init__(self, value: str = "defects", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure the id columns in both dataframes are of string type for consistent sorting
        y_true[y_true.columns[0]] = y_true[y_true.columns[0]].astype(str)
        y_pred[y_pred.columns[0]] = y_pred[y_pred.columns[0]].astype(str)
        
        # Sort both dataframes by the id column (first column) to align predictions with ground truth
        y_true = y_true.sort_values(by=y_true.columns[0]).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=y_pred.columns[0]).reset_index(drop=True)
        
        # Calculate Area Under the ROC Curve
        score = roc_auc_score(y_true[self.value].astype(int), y_pred[self.value].values)
        return score

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(
                f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows."
            )

        # Convert first column (id) to string for consistent comparison
        submission[submission.columns[0]] = submission[submission.columns[0]].astype(str)
        ground_truth[ground_truth.columns[0]] = ground_truth[ground_truth.columns[0]].astype(str)
        
        # Sort both submission and ground truth by the id column
        submission = submission.sort_values(by=submission.columns[0]).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=ground_truth.columns[0]).reset_index(drop=True)

        # Check if their id columns match exactly
        if (submission[submission.columns[0]].values != ground_truth[ground_truth.columns[0]].values).any():
            raise InvalidSubmissionError(
                "Values in the id column do not match between submission and ground truth. Please ensure the first column values are identical."
            )

        sub_cols = set(submission.columns)
        true_cols = set(ground_truth.columns)

        missing_cols = true_cols - sub_cols
        extra_cols = sub_cols - true_cols

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."