from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class WalmartRecruitingStoreSalesForecastingMetrics(CompetitionMetrics):
    """
    Metric class for the Walmart Recruiting Store Sales Forecasting competition.
    Evaluation is based on Weighted Mean Absolute Error (WMAE):
    
    WMAE = (sum_i w_i * |y_i - ŷ_i|) / (sum_i w_i)
    
    where w_i = 5 if the week is a holiday week (IsHoliday is True), 1 otherwise.
    """
    def __init__(self, value: str = "Weekly_Sales", higher_is_better: bool = False):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure Id columns are string type
        id_col_true = y_true.columns[0]
        id_col_pred = y_pred.columns[0]
        y_true[id_col_true] = y_true[id_col_true].astype(str)
        y_pred[id_col_pred] = y_pred[id_col_pred].astype(str)
        
        # Sort both DataFrames by Id
        y_true = y_true.sort_values(by=id_col_true).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=id_col_pred).reset_index(drop=True)
        
        # Compute absolute errors between the true and predicted Weekly_Sales
        errors = np.abs(y_true[self.value] - y_pred[self.value])
        
        # Determine weights: 5 if holiday, 1 otherwise.
        # We expect the ground truth DataFrame to have an "IsHoliday" column.
        if "IsHoliday" in y_true.columns:
            weights = np.where(y_true["IsHoliday"] == True, 5, 1)
        else:
            weights = np.ones(len(y_true))
        
        total_weight = np.sum(weights)
        if total_weight == 0:
            raise ValueError("Sum of weights is zero, cannot compute WMAE.")
        
        wmae = np.sum(weights * errors) / total_weight
        return wmae

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        
        # For this competition, the submission must contain exactly two columns: "Id" and "Weekly_Sales".
        required_columns = {"Id", "Weekly_Sales"}
        submission_columns = set(submission.columns)
        if submission_columns != required_columns:
            missing_cols = required_columns - submission_columns
            extra_cols = submission_columns - required_columns
            error_msgs = []
            if missing_cols:
                error_msgs.append(f"Missing required columns: {', '.join(missing_cols)}.")
            if extra_cols:
                error_msgs.append(f"Extra unexpected columns: {', '.join(extra_cols)}.")
            raise InvalidSubmissionError(" ".join(error_msgs))
        
        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}).")
        
        # Convert Id column to string type and sort both DataFrames by Id
        submission["Id"] = submission["Id"].astype(str)
        ground_truth["Id"] = ground_truth["Id"].astype(str)
        submission_sorted = submission.sort_values(by="Id").reset_index(drop=True)
        ground_truth_sorted = ground_truth.sort_values(by="Id").reset_index(drop=True)
        
        # Check if Id columns match between submission and ground truth
        if not (submission_sorted["Id"].values == ground_truth_sorted["Id"].values).all():
            raise InvalidSubmissionError("The Id values in submission do not match those in the ground truth.")
        
        return "Submission is valid."