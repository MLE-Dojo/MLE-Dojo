from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError
from sklearn.metrics import matthews_corrcoef

class NflPlayerContactDetectionMetrics(CompetitionMetrics):
    """Metric class for NFL Player Contact Detection competition using Matthews Correlation Coefficient"""
    def __init__(self, value: str = "contact", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure contact_id is treated as string
        y_true["contact_id"] = y_true["contact_id"].astype(str)
        y_pred["contact_id"] = y_pred["contact_id"].astype(str)
        # Sort by the id column (contact_id)
        y_true = y_true.sort_values(by="contact_id").reset_index(drop=True)
        y_pred = y_pred.sort_values(by="contact_id").reset_index(drop=True)
        return matthews_corrcoef(y_true[self.value], y_pred[self.value])

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Ensure the id column is present and correctly typed
        if "contact_id" not in submission.columns or "contact_id" not in ground_truth.columns:
            raise InvalidSubmissionError("Both submission and ground truth must contain the 'contact_id' column.")

        submission["contact_id"] = submission["contact_id"].astype(str)
        ground_truth["contact_id"] = ground_truth["contact_id"].astype(str)
        # Sort by the id column
        submission = submission.sort_values(by="contact_id").reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by="contact_id").reset_index(drop=True)

        # Check if contact_id values are identical
        if not (submission["contact_id"].values == ground_truth["contact_id"].values).all():
            raise InvalidSubmissionError("The 'contact_id' values do not match between submission and ground truth. Please ensure they are identical.")

        required_columns = {"contact_id", "contact"}
        submission_cols = set(submission.columns)
        ground_truth_cols = set(ground_truth.columns)

        missing_cols = required_columns - submission_cols
        extra_cols = submission_cols - ground_truth_cols

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."