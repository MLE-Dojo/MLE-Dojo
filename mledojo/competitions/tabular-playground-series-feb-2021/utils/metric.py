from typing import Any
import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class TabularPlaygroundSeriesFeb2021Metrics(CompetitionMetrics):
    """Metric class for Tabular Playground Series Feb 2021 competition using RMSE."""
    def __init__(self, value: str = "target", higher_is_better: bool = False):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Convert 'id' columns to string type to ensure proper sorting
        y_true[y_true.columns[0]] = y_true[y_true.columns[0]].astype(str)
        y_pred[y_pred.columns[0]] = y_pred[y_pred.columns[0]].astype(str)
        
        # Sort both dataframes based on the id column (first column)
        y_true = y_true.sort_values(by=y_true.columns[0]).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=y_pred.columns[0]).reset_index(drop=True)
        
        # Calculate RMSE between the true and predicted target columns
        rmse = math.sqrt(mean_squared_error(y_true[self.value], y_pred[self.value]))
        return rmse

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

        # Ensure the first column (id) is of string type
        submission[submission.columns[0]] = submission[submission.columns[0]].astype(str)
        ground_truth[ground_truth.columns[0]] = ground_truth[ground_truth.columns[0]].astype(str)
        
        # Sort both submission and ground truth by the id column
        submission = submission.sort_values(by=submission.columns[0]).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=ground_truth.columns[0]).reset_index(drop=True)

        # Check if the id columns are identical
        if not np.array_equal(submission[submission.columns[0]].values, ground_truth[ground_truth.columns[0]].values):
            raise InvalidSubmissionError(
                "First column values (id) do not match between submission and ground truth. Please ensure the first column values are identical."
            )

        # Check for required columns
        required_columns = {submission.columns[0], self.value}
        sub_cols = set(submission.columns)
        true_cols = set(ground_truth.columns)

        missing_cols = required_columns - sub_cols
        extra_cols = sub_cols - required_columns
            
        if missing_cols:
            raise InvalidSubmissionError(
                f"Missing required columns in submission: {', '.join(missing_cols)}."
            )
        if extra_cols:
            raise InvalidSubmissionError(
                f"Extra unexpected columns found in submission: {', '.join(extra_cols)}."
            )

        return "Submission is valid."