from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class Covid19GlobalForecastingWeek2Metrics(CompetitionMetrics):
    """
    Metric class for the covid19-global-forecasting-week-2 competition.
    Evaluation is based on the column-wise Root Mean Squared Logarithmic Error (RMSLE)
    for the 'ConfirmedCases' and 'Fatalities' columns.
    The final score is the mean RMSLE computed over these two columns.
    """
    def __init__(self, value: str = "RMSLE", higher_is_better: bool = False):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure the identifier column (ForecastId) is of string type
        id_col = y_true.columns[0]
        y_true[id_col] = y_true[id_col].astype(str)
        y_pred[id_col] = y_pred[id_col].astype(str)

        # Sort both DataFrames by the identifier column
        y_true = y_true.sort_values(by=id_col).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=id_col).reset_index(drop=True)
        
        # Check required columns
        required_columns = ['ConfirmedCases', 'Fatalities']
        for col in required_columns:
            if col not in y_true.columns or col not in y_pred.columns:
                raise InvalidSubmissionError(f"Missing required column: {col}")
        
        rmsle_scores = []
        for col in required_columns:
            preds = y_pred[col].values.astype(float)
            actuals = y_true[col].values.astype(float)
            # Calculate RMSLE for this column
            if len(preds) != len(actuals):
                raise InvalidSubmissionError("Number of predictions and ground truth values do not match.")
            # Compute the logarithms after adding 1
            log_preds = np.log1p(preds)
            log_actuals = np.log1p(actuals)
            # Calculate mean squared logarithmic error and then the root
            msle = np.mean((log_preds - log_actuals) ** 2)
            rmsle = np.sqrt(msle)
            rmsle_scores.append(rmsle)
        
        # Final score is the mean of both RMSLE values
        return float(np.mean(rmsle_scores))

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Convert identifier column values to string type for consistent sorting and comparison.
        id_col = ground_truth.columns[0]
        submission[id_col] = submission[id_col].astype(str)
        ground_truth[id_col] = ground_truth[id_col].astype(str)
        # Sort both DataFrames by the identifier column.
        submission = submission.sort_values(by=id_col).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=id_col).reset_index(drop=True)

        # Check that identifier columns match exactly
        if (submission[id_col].values != ground_truth[id_col].values).any():
            raise InvalidSubmissionError("Identifier column values do not match between submission and ground truth. Please ensure that the ForecastId column values are identical and in the same order.")

        # Check if required columns exist and match exactly
        required_columns = {id_col, 'ConfirmedCases', 'Fatalities'}
        submission_cols = set(submission.columns)
        ground_truth_cols = set(ground_truth.columns)

        missing_cols = ground_truth_cols - submission_cols
        extra_cols = submission_cols - ground_truth_cols

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."