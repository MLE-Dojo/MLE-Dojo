from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class Covid19GlobalForecastingWeek4Metrics(CompetitionMetrics):
    """
    Metric class for COVID19 Global Forecasting Week 4 competition using column-wise RMSLE.
    The final score is the mean of the RMSLE for ConfirmedCases and Fatalities.
    """
    def __init__(self, value: str = "mean_rmsle", higher_is_better: bool = False):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure the ForecastId columns are of type string for proper sorting and comparison
        y_true["ForecastId"] = y_true["ForecastId"].astype(str)
        y_pred["ForecastId"] = y_pred["ForecastId"].astype(str)
        
        # Sort both dataframes by ForecastId
        y_true = y_true.sort_values(by="ForecastId").reset_index(drop=True)
        y_pred = y_pred.sort_values(by="ForecastId").reset_index(drop=True)
        
        # Define the columns for evaluation
        eval_columns = ["ConfirmedCases", "Fatalities"]
        rmsle_list = []
        
        for col in eval_columns:
            # Convert values to float for safety
            true_vals = y_true[col].astype(float).values
            pred_vals = y_pred[col].astype(float).values
            # Compute RMSLE for current column
            log_diff = np.log1p(pred_vals) - np.log1p(true_vals)
            rmsle = np.sqrt(np.mean(np.square(log_diff)))
            rmsle_list.append(rmsle)
        
        # Final score is the average of the RMSLEs over both columns
        final_score = np.mean(rmsle_list)
        return final_score

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Convert ForecastId column to string type for both submission and ground truth
        if "ForecastId" not in submission.columns or "ForecastId" not in ground_truth.columns:
            raise InvalidSubmissionError("Both submission and ground truth must contain 'ForecastId' column.")
            
        submission["ForecastId"] = submission["ForecastId"].astype(str)
        ground_truth["ForecastId"] = ground_truth["ForecastId"].astype(str)
        
        # Sort both submission and ground truth by ForecastId
        submission = submission.sort_values(by="ForecastId").reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by="ForecastId").reset_index(drop=True)

        # Check if ForecastId values are identical
        if not np.array_equal(submission["ForecastId"].values, ground_truth["ForecastId"].values):
            raise InvalidSubmissionError("ForecastId values do not match between submission and ground truth. Please ensure the ForecastId values are identical and in the same order.")

        required_columns = {"ForecastId", "ConfirmedCases", "Fatalities"}
        submission_columns = set(submission.columns)
        ground_truth_columns = set(ground_truth.columns)

        missing_cols = required_columns - submission_columns
        extra_cols = submission_columns - required_columns
            
        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."