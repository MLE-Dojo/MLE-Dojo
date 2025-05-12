from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError
from sklearn.metrics import accuracy_score

class SpaceshipTitanicMetrics(CompetitionMetrics):
    """Metric class for Spaceship Titanic competition using Classification Accuracy"""

    def __init__(self, value: str = "Transported", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Convert the PassengerId column to string type and sort by it
        id_col = y_true.columns[0]
        y_true[id_col] = y_true[id_col].astype(str)
        y_pred[id_col] = y_pred[id_col].astype(str)
        y_true = y_true.sort_values(by=id_col).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=id_col).reset_index(drop=True)
        
        # Compute classification accuracy
        return accuracy_score(y_true[self.value], y_pred[self.value])

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Convert the PassengerId column to string type
        id_col = submission.columns[0]
        submission[id_col] = submission[id_col].astype(str)
        ground_truth[id_col] = ground_truth[id_col].astype(str)
        # Sort the submission and ground truth by the PassengerId
        submission = submission.sort_values(by=id_col).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=id_col).reset_index(drop=True)

        # Check if PassengerId values are identical
        if (submission[id_col].values != ground_truth[id_col].values).any():
            raise InvalidSubmissionError("PassengerId values do not match between submission and ground truth. Please ensure the PassengerId values are identical.")

        submission_cols = set(submission.columns)
        truth_cols = set(ground_truth.columns)

        missing_cols = truth_cols - submission_cols
        extra_cols = submission_cols - truth_cols

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."