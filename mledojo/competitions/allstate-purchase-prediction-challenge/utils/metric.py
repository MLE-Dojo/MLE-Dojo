from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class AllstatePurchasePredictionChallengeMetrics(CompetitionMetrics):
    """Metric class for Allstate Purchase Prediction Challenge using exact match percentage"""
    def __init__(self, value: str = "plan", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure 'customer_ID' column is of string type and sort based on it
        y_true["customer_ID"] = y_true["customer_ID"].astype(str)
        y_pred["customer_ID"] = y_pred["customer_ID"].astype(str)
        y_true = y_true.sort_values(by="customer_ID").reset_index(drop=True)
        y_pred = y_pred.sort_values(by="customer_ID").reset_index(drop=True)
        
        # Calculate exact match accuracy: percent of customers for whom the predicted plan matches the true plan exactly.
        correct_predictions = (y_true[self.value] == y_pred[self.value]).sum()
        score = (correct_predictions / len(y_true)) * 100.0
        return score

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Ensure the 'customer_ID' column is of string type and sort both submission and ground truth by it
        if "customer_ID" not in submission.columns or "customer_ID" not in ground_truth.columns:
            raise InvalidSubmissionError("Both submission and ground truth must contain 'customer_ID' column.")

        submission["customer_ID"] = submission["customer_ID"].astype(str)
        ground_truth["customer_ID"] = ground_truth["customer_ID"].astype(str)

        submission = submission.sort_values(by="customer_ID").reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by="customer_ID").reset_index(drop=True)

        # Check if 'customer_ID' column values are identical between submission and ground truth
        if not (submission["customer_ID"].values == ground_truth["customer_ID"].values).all():
            raise InvalidSubmissionError("The 'customer_ID' values do not match between submission and ground truth. Please ensure they are in the same order and identical.")

        sub_cols = set(submission.columns)
        true_cols = set(ground_truth.columns)
        missing_cols = true_cols - sub_cols
        extra_cols = sub_cols - true_cols

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."