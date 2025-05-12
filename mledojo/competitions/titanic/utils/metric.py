from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class TitanicMetrics(CompetitionMetrics):
    """Metric class for the Titanic competition using Accuracy as metric."""
    def __init__(self, value: str = "Survived", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure PassengerId is string for proper sorting
        y_true["PassengerId"] = y_true["PassengerId"].astype(str)
        y_pred["PassengerId"] = y_pred["PassengerId"].astype(str)
        # Sort both DataFrames by PassengerId
        y_true = y_true.sort_values(by="PassengerId").reset_index(drop=True)
        y_pred = y_pred.sort_values(by="PassengerId").reset_index(drop=True)
        # Compute accuracy: percentage of correct predictions
        accuracy = np.mean(y_true[self.value] == y_pred[self.value])
        return accuracy

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(
                f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)})."
            )

        expected_columns = {"PassengerId", "Survived"}
        submission_cols = set(submission.columns)
        ground_truth_cols = set(ground_truth.columns)

        missing_cols = expected_columns - submission_cols
        extra_cols = submission_cols - expected_columns

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        # Convert PassengerId to string for both DataFrames and sort by it
        submission["PassengerId"] = submission["PassengerId"].astype(str)
        ground_truth["PassengerId"] = ground_truth["PassengerId"].astype(str)

        submission = submission.sort_values(by="PassengerId").reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by="PassengerId").reset_index(drop=True)

        if not (submission["PassengerId"].values == ground_truth["PassengerId"].values).all():
            raise InvalidSubmissionError("PassengerId values do not match between submission and ground truth. Please ensure the PassengerId order is identical.")

        return "Submission is valid."