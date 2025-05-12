from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class GrupoBimboInventoryDemandMetrics(CompetitionMetrics):
    """
    Metric class for Grupo Bimbo Inventory Demand competition using Root Mean Squared Logarithmic Error (RMSLE).
    Lower RMSLE indicates a better score.
    """
    def __init__(self, value: str = "Demanda_uni_equil", higher_is_better: bool = False):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure that the 'id' columns are of string type and sort by 'id'
        y_true["id"] = y_true["id"].astype(str)
        y_pred["id"] = y_pred["id"].astype(str)
        y_true = y_true.sort_values(by="id").reset_index(drop=True)
        y_pred = y_pred.sort_values(by="id").reset_index(drop=True)
        
        # Calculate RMSLE: sqrt(mean((log(pred+1) - log(true+1))^2))
        pred = y_pred[self.value].values
        true = y_true[self.value].values
        
        # To ensure numerical stability, clip predictions and true values to be non-negative
        pred = np.clip(pred, a_min=0, a_max=None)
        true = np.clip(true, a_min=0, a_max=None)
        
        rmsle = np.sqrt(np.mean((np.log1p(pred) - np.log1p(true)) ** 2))
        return rmsle

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(
                f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Check if required columns exist in both submission and ground truth
        required_columns = {"id", self.value}
        submission_cols = set(submission.columns)
        ground_truth_cols = set(ground_truth.columns)

        missing_cols = required_columns - submission_cols
        extra_cols = submission_cols - required_columns
        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        # Convert 'id' column to string type and sort both DataFrames by 'id'
        submission["id"] = submission["id"].astype(str)
        ground_truth["id"] = ground_truth["id"].astype(str)
        submission_sorted = submission.sort_values(by="id").reset_index(drop=True)
        ground_truth_sorted = ground_truth.sort_values(by="id").reset_index(drop=True)

        # Check if the 'id' columns are identical
        if not (submission_sorted["id"].values == ground_truth_sorted["id"].values).all():
            raise InvalidSubmissionError("The 'id' column values do not match between submission and ground truth. Please ensure they are identical.")

        return "Submission is valid."