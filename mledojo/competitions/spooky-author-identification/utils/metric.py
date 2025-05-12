from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError


class SpookyAuthorIdentificationMetrics(CompetitionMetrics):
    """Metric class for Spooky Author Identification competition using Log Loss"""

    def __init__(self, value: str = "EAP,HPL,MWS", higher_is_better: bool = False):
        super().__init__(higher_is_better)
        self.value = value.split(",")

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Sort both dataframes by first column before calculating score
        y_true = y_true.sort_values(by=y_true.columns[0]).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=y_pred.columns[0]).reset_index(drop=True)
        y_true = y_true[self.value]
        y_pred = y_pred[self.value]
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        log_loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return log_loss

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError(
                "Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame."
            )
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError(
                "Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame."
            )

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(
                f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows."
            )

        # Sort the submission and ground truth by the first column
        submission = submission.sort_values(by=submission.columns[0])
        ground_truth = ground_truth.sort_values(by=ground_truth.columns[0])

        # Check if first columns are identical
        if (submission[submission.columns[0]].values != (ground_truth[ground_truth.columns[0]].values)).any():
            raise InvalidSubmissionError(
                "First column values do not match between submission and ground truth. Please ensure the first column values are identical."
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
