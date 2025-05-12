from typing import Any
import numpy as np
import pandas as pd
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class Covid19GlobalForecastingWeek1Metrics(CompetitionMetrics):
    """Metric class for the covid19-global-forecasting-week-1 competition using RMSLE for ConfirmedCases and Fatalities."""
    def __init__(self, value: str = "ConfirmedCases", higher_is_better: bool = False):
        super().__init__(higher_is_better)
        self.value = value
        self.required_columns = {'ForecastId', 'ConfirmedCases', 'Fatalities'}

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Convert the first column (ForecastId) to string and sort both DataFrames by it
        id_col = y_true.columns[0]
        y_true[id_col] = y_true[id_col].astype(str)
        y_pred[id_col] = y_pred[id_col].astype(str)
        y_true = y_true.sort_values(by=id_col).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=id_col).reset_index(drop=True)

        # Ensure the prediction columns are of float type
        confirmed_true = y_true["ConfirmedCases"].astype(float)
        confirmed_pred = y_pred["ConfirmedCases"].astype(float)
        fatalities_true = y_true["Fatalities"].astype(float)
        fatalities_pred = y_pred["Fatalities"].astype(float)

        # Calculate RMSLE for ConfirmedCases
        rmsle_confirmed = np.sqrt(
            np.mean((np.log1p(confirmed_pred) - np.log1p(confirmed_true)) ** 2)
        )

        # Calculate RMSLE for Fatalities
        rmsle_fatalities = np.sqrt(
            np.mean((np.log1p(fatalities_pred) - np.log1p(fatalities_true)) ** 2)
        )

        # Return the average RMSLE over both columns
        return (rmsle_confirmed + rmsle_fatalities) / 2.0

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError(
                "Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame."
            )
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError(
                "Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame."
            )

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(
                f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows."
            )

        id_col = ground_truth.columns[0]
        submission[id_col] = submission[id_col].astype(str)
        ground_truth[id_col] = ground_truth[id_col].astype(str)
        submission = submission.sort_values(by=id_col).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=id_col).reset_index(drop=True)

        # Check if the ForecastId values match between submission and ground truth
        if (submission[id_col].values != ground_truth[id_col].values).any():
            raise InvalidSubmissionError(
                "ForecastId values do not match between submission and ground truth. Please ensure the ForecastId values are identical."
            )

        submission_cols = set(submission.columns)
        missing_cols = self.required_columns - submission_cols
        extra_cols = submission_cols - self.required_columns

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."