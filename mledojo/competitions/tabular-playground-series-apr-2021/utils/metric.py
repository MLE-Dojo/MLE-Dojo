from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError
from sklearn.metrics import accuracy_score

class TabularPlaygroundSeriesApr2021Metrics(CompetitionMetrics):
    """Metric class for Tabular Playground Series Apr 2021 competition using Accuracy."""
    def __init__(self, value: str = "Survived", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Convert ID column to string type for both true and predicted dataframes
        id_col_true = y_true.columns[0]
        id_col_pred = y_pred.columns[0]
        y_true[id_col_true] = y_true[id_col_true].astype(str)
        y_pred[id_col_pred] = y_pred[id_col_pred].astype(str)
        
        # Sort the dataframes by the id column
        y_true = y_true.sort_values(by=id_col_true).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=id_col_pred).reset_index(drop=True)
        
        # Calculate and return accuracy score
        return accuracy_score(y_true[self.value], y_pred[self.value])

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        
        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(
                f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows."
            )
        
        # Convert the id column to string type
        id_col_sub = submission.columns[0]
        id_col_true = ground_truth.columns[0]
        submission[id_col_sub] = submission[id_col_sub].astype(str)
        ground_truth[id_col_true] = ground_truth[id_col_true].astype(str)
        
        # Sort both DataFrames by the id column
        submission_sorted = submission.sort_values(by=id_col_sub).reset_index(drop=True)
        ground_truth_sorted = ground_truth.sort_values(by=id_col_true).reset_index(drop=True)
        
        # Check if the id columns are identical
        if not (submission_sorted[id_col_sub].values == ground_truth_sorted[id_col_true].values).all():
            raise InvalidSubmissionError("First column values do not match between submission and ground truth. Please ensure the 'PassengerId' column values are identical.")
        
        # Check for required columns
        required_cols = set(ground_truth.columns)
        submission_cols = set(submission.columns)
        
        missing_cols = required_cols - submission_cols
        extra_cols = submission_cols - required_cols
        
        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")
        
        return "Submission is valid."