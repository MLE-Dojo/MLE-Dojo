from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError
from sklearn.metrics import log_loss

class QuoraQuestionPairsMetrics(CompetitionMetrics):
    """Metric class for Quora Question Pairs competition using Log Loss."""
    def __init__(self, value: str = "is_duplicate", higher_is_better: bool = False):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure the id column 'test_id' is of type string for both y_true and y_pred
        y_true["test_id"] = y_true["test_id"].astype(str)
        y_pred["test_id"] = y_pred["test_id"].astype(str)
        # Sort both dataframes by the 'test_id' column
        y_true = y_true.sort_values(by="test_id").reset_index(drop=True)
        y_pred = y_pred.sort_values(by="test_id").reset_index(drop=True)
        # Calculate log loss between ground truth and prediction probabilities
        return log_loss(y_true[self.value].values, y_pred[self.value].values)

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        required_columns = {"test_id", self.value}
        submission_columns = set(submission.columns)
        ground_truth_columns = set(ground_truth.columns)

        if not required_columns.issubset(submission_columns):
            missing = required_columns - submission_columns
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing)}.")
        if not required_columns.issubset(ground_truth_columns):
            missing = required_columns - ground_truth_columns
            raise InvalidSubmissionError(f"Missing required columns in ground truth: {', '.join(missing)}.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Convert and sort by 'test_id'
        submission["test_id"] = submission["test_id"].astype(str)
        ground_truth["test_id"] = ground_truth["test_id"].astype(str)
        submission = submission.sort_values(by="test_id").reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by="test_id").reset_index(drop=True)

        # Check if test_id columns match exactly
        if not np.array_equal(submission["test_id"].values, ground_truth["test_id"].values):
            raise InvalidSubmissionError("The 'test_id' values do not match between submission and ground truth. Please ensure they are identical and in the same order.")

        extra_cols = submission_columns - ground_truth_columns
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."