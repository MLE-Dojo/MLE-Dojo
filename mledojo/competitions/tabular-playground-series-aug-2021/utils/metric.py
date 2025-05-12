from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError
from sklearn.metrics import mean_squared_error

class TabularPlaygroundSeriesAug2021Metrics(CompetitionMetrics):
    """Metric class for Tabular Playground Series Aug 2021 competition using RMSE"""
    def __init__(self, value: str = "loss", higher_is_better: bool = False):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure the identifier column (first column, assumed to be 'id') is of string type
        y_true[y_true.columns[0]] = y_true[y_true.columns[0]].astype(str)
        y_pred[y_pred.columns[0]] = y_pred[y_pred.columns[0]].astype(str)
        
        # Sort both DataFrames by the id column
        y_true = y_true.sort_values(by=y_true.columns[0]).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=y_pred.columns[0]).reset_index(drop=True)
        
        # Calculate the root mean squared error (RMSE)
        mse = mean_squared_error(y_true[self.value], y_pred[self.value])
        rmse = np.sqrt(mse)
        return rmse

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
            
        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")
        
        # Ensure the first column (id) is of string type in both DataFrames
        submission[submission.columns[0]] = submission[submission.columns[0]].astype(str)
        ground_truth[ground_truth.columns[0]] = ground_truth[ground_truth.columns[0]].astype(str)
        
        # Sort the submission and ground truth by the identifier (first) column
        submission = submission.sort_values(by=submission.columns[0]).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=ground_truth.columns[0]).reset_index(drop=True)
        
        # Check if identifier columns match exactly
        if not np.array_equal(submission[submission.columns[0]].values, ground_truth[ground_truth.columns[0]].values):
            raise InvalidSubmissionError("Identifier column values do not match between submission and ground truth. Please ensure the first column values are identical and in the same order.")
        
        sub_cols = set(submission.columns)
        true_cols = set(ground_truth.columns)
        
        missing_cols = true_cols - sub_cols
        extra_cols = sub_cols - true_cols
        
        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")
        
        return "Submission is valid."