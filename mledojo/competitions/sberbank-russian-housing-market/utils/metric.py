from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class SberbankRussianHousingMarketMetrics(CompetitionMetrics):
    """Metric class for Sberbank Russian Housing Market competition using RMSLE"""
    def __init__(self, value: str = "price_doc", higher_is_better: bool = False):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Convert the first column (id) to string type for both ground truth and predictions
        y_true[y_true.columns[0]] = y_true[y_true.columns[0]].astype(str)
        y_pred[y_pred.columns[0]] = y_pred[y_pred.columns[0]].astype(str)
        # Sort both by the id column
        y_true = y_true.sort_values(by=y_true.columns[0]).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=y_pred.columns[0]).reset_index(drop=True)
        
        # Compute Root Mean Squared Logarithmic Error (RMSLE)
        # Using np.log1p to avoid issues with zero values
        log_true = np.log1p(y_true[self.value])
        log_pred = np.log1p(y_pred[self.value])
        rmsle = np.sqrt(np.mean((log_pred - log_true) ** 2))
        return rmsle

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(
                f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows."
            )

        # Convert the first column (id) to string type for both submission and ground truth
        submission[submission.columns[0]] = submission[submission.columns[0]].astype(str)
        ground_truth[ground_truth.columns[0]] = ground_truth[ground_truth.columns[0]].astype(str)
        # Sort both by the id column
        submission = submission.sort_values(by=submission.columns[0]).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=ground_truth.columns[0]).reset_index(drop=True)

        # Check if the id columns match
        if not (submission[submission.columns[0]].values == ground_truth[ground_truth.columns[0]].values).all():
            raise InvalidSubmissionError("The id values in submission do not match those in ground truth. Please ensure they are identical and in the same order.")

        required_columns = {ground_truth.columns[0], self.value}
        submission_columns = set(submission.columns)

        missing_cols = required_columns - submission_columns
        extra_cols = submission_columns - required_columns

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."