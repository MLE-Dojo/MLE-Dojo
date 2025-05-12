from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError
from sklearn.metrics import cohen_kappa_score

class PrudentialLifeInsuranceAssessmentMetrics(CompetitionMetrics):
    """Metric class for Prudential Life Insurance Assessment competition using Quadratic Weighted Kappa."""
    def __init__(self, value: str = "Response", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure the Id column exists
        if "Id" not in y_true.columns or "Id" not in y_pred.columns:
            raise InvalidSubmissionError("Both y_true and y_pred must contain 'Id' column for proper sorting.")
        
        # Convert 'Id' column to string type
        y_true["Id"] = y_true["Id"].astype(str)
        y_pred["Id"] = y_pred["Id"].astype(str)
        
        # Sort by 'Id' column
        y_true = y_true.sort_values(by="Id").reset_index(drop=True)
        y_pred = y_pred.sort_values(by="Id").reset_index(drop=True)
        
        # Evaluate quadratic weighted kappa using the 'Response' column
        return cohen_kappa_score(y_true[self.value], y_pred[self.value], weights='quadratic')

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        if "Id" not in submission.columns or "Id" not in ground_truth.columns:
            raise InvalidSubmissionError("Both submission and ground truth must contain the 'Id' column.")

        # Convert 'Id' column to string type
        submission["Id"] = submission["Id"].astype(str)
        ground_truth["Id"] = ground_truth["Id"].astype(str)
        
        # Sort the submission and ground truth by the 'Id' column
        submission = submission.sort_values(by="Id").reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by="Id").reset_index(drop=True)

        # Check if Id column values are identical
        if not (submission["Id"].values == ground_truth["Id"].values).all():
            raise InvalidSubmissionError("Id values do not match between submission and ground truth. Please ensure the 'Id' column values are identical and in the same order.")

        sub_cols = set(submission.columns)
        gt_cols = set(ground_truth.columns)

        missing_cols = gt_cols - sub_cols
        extra_cols = sub_cols - gt_cols

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."