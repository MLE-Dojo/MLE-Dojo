from typing import Any
import pandas as pd
import numpy as np
from sklearn.metrics import median_absolute_error
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class PlaygroundSeriesS3e25Metrics(CompetitionMetrics):
    """Metric class for playground-series-s3e25 using Median Absolute Error (MedAE)"""
    def __init__(self, value: str = "Hardness", higher_is_better: bool = False):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Convert the first column (id) to string type to ensure proper sorting
        y_true[y_true.columns[0]] = y_true[y_true.columns[0]].astype(str)
        y_pred[y_pred.columns[0]] = y_pred[y_pred.columns[0]].astype(str)
        # Sort the dataframes by id column
        y_true = y_true.sort_values(by=y_true.columns[0]).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=y_pred.columns[0]).reset_index(drop=True)
        # Compute the Median Absolute Error using the Hardness column
        score = median_absolute_error(y_true[self.value], y_pred[self.value])
        return score

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Convert the first column (id) to string type for both submission and ground truth
        submission[submission.columns[0]] = submission[submission.columns[0]].astype(str)
        ground_truth[ground_truth.columns[0]] = ground_truth[ground_truth.columns[0]].astype(str)
        # Sort the dataframes by the first column (id)
        submission = submission.sort_values(by=submission.columns[0]).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=ground_truth.columns[0]).reset_index(drop=True)

        # Check if first column values (id) are identical
        if not np.array_equal(submission[submission.columns[0]].values, ground_truth[ground_truth.columns[0]].values):
            raise InvalidSubmissionError("First column values do not match between submission and ground truth. Please ensure the id values are identical.")

        # Check for required columns
        required_columns = set(ground_truth.columns)
        submission_columns = set(submission.columns)

        missing_cols = required_columns - submission_columns
        extra_cols = submission_columns - required_columns

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."