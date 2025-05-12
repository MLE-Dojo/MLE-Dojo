from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class TabularPlaygroundSeriesSep2022Metrics(CompetitionMetrics):
    def __init__(self, value: str = "num_sold", higher_is_better: bool = False):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Convert the first column (row_id) to string type for both dataframes
        y_true[y_true.columns[0]] = y_true[y_true.columns[0]].astype(str)
        y_pred[y_pred.columns[0]] = y_pred[y_pred.columns[0]].astype(str)

        # Sort both dataframes by the row_id
        y_true = y_true.sort_values(by=y_true.columns[0]).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=y_pred.columns[0]).reset_index(drop=True)

        # Extract the actual and predicted values
        actual = y_true[self.value].to_numpy().astype(np.float64)
        forecast = y_pred[self.value].to_numpy().astype(np.float64)

        # Compute SMAPE: Use 0 when both actual and forecast are 0
        denominator = np.abs(actual) + np.abs(forecast)
        diff = np.abs(forecast - actual)
        smape = np.mean(np.where(denominator == 0, 0, 200 * diff / denominator))
        return smape

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(
                f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). "
                "Please ensure both have the same number of rows."
            )

        # Convert row_id to string type for proper sorting and comparison
        submission[submission.columns[0]] = submission[submission.columns[0]].astype(str)
        ground_truth[ground_truth.columns[0]] = ground_truth[ground_truth.columns[0]].astype(str)

        # Sort the submission and ground truth by the first column (row_id)
        submission = submission.sort_values(by=submission.columns[0])
        ground_truth = ground_truth.sort_values(by=ground_truth.columns[0])

        # Check if the row_id values are identical between submission and ground truth
        if not (submission[submission.columns[0]].values == ground_truth[ground_truth.columns[0]].values).all():
            raise InvalidSubmissionError("First column values do not match between submission and ground truth. Please ensure the first column values are identical.")

        required_cols = {submission.columns[0], self.value}
        submission_cols = set(submission.columns)
        ground_truth_cols = set(ground_truth.columns)

        missing_cols = required_cols - submission_cols
        extra_cols = submission_cols - ground_truth_cols

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."