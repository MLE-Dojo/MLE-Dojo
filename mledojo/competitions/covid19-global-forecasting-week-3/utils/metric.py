from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class Covid19GlobalForecastingWeek3Metrics(CompetitionMetrics):
    """Metric class for Kaggle's COVID-19 Global Forecasting Week 3 competition using column-wise RMSLE."""
    def __init__(self, value: str = "ForecastId", higher_is_better: bool = False):
        super().__init__(higher_is_better)
        self.value = value  # this is the id column

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure that the id column (ForecastId) is of string type and sort by it
        id_col = self.value
        y_true[id_col] = y_true[id_col].astype(str)
        y_pred[id_col] = y_pred[id_col].astype(str)
        y_true = y_true.sort_values(by=id_col).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=id_col).reset_index(drop=True)
        
        required_columns = ["ConfirmedCases", "Fatalities"]
        for col in required_columns:
            if col not in y_true.columns or col not in y_pred.columns:
                raise InvalidSubmissionError(f"Missing required column: {col}")

        # Compute RMSLE for each required column
        rmsle_values = []
        for col in required_columns:
            # Ensure predictions and actuals are non-negative
            if (y_pred[col] < 0).any() or (y_true[col] < 0).any():
                raise InvalidSubmissionError(f"Negative values found in column: {col}")

            # Calculate the log transformed errors
            log_pred = np.log(y_pred[col].values + 1)
            log_true = np.log(y_true[col].values + 1)
            squared_errors = (log_pred - log_true) ** 2
            mean_squared_error = np.mean(squared_errors)
            rmsle = np.sqrt(mean_squared_error)
            rmsle_values.append(rmsle)
        
        # The final score is the mean of the RMSLE values for ConfirmedCases and Fatalities
        return np.mean(rmsle_values)

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        id_col = self.value
        # Convert the id column to string type
        submission[id_col] = submission[id_col].astype(str)
        ground_truth[id_col] = ground_truth[id_col].astype(str)
        # Sort by the id column
        submission = submission.sort_values(by=id_col).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=id_col).reset_index(drop=True)

        # Check if id columns are identical between submission and ground truth
        if not np.array_equal(submission[id_col].values, ground_truth[id_col].values):
            raise InvalidSubmissionError("First column values (ForecastId) do not match between submission and ground truth. Please ensure the ForecastId values are identical and in the same order.")

        # Define the required columns
        required_cols = {"ForecastId", "ConfirmedCases", "Fatalities"}
        submission_cols = set(submission.columns)
        ground_truth_cols = set(ground_truth.columns)

        missing_cols = required_cols - submission_cols
        extra_cols = submission_cols - required_cols

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."