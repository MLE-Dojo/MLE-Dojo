from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError
from sklearn.metrics import roc_auc_score

class PlaygroundSeriesS3e3Metrics(CompetitionMetrics):
    """Metric class for Playground Series S3E3 competition using Area Under ROC Curve."""
    def __init__(self, value: str = "Attrition", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure the id column (EmployeeNumber) is a string for proper sorting.
        id_column_true = y_true.columns[0]
        id_column_pred = y_pred.columns[0]
        y_true[id_column_true] = y_true[id_column_true].astype(str)
        y_pred[id_column_pred] = y_pred[id_column_pred].astype(str)
        
        # Sort by the id column to align both dataframes.
        y_true = y_true.sort_values(by=id_column_true).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=id_column_pred).reset_index(drop=True)
        
        # Compute and return the area under the ROC curve.
        return roc_auc_score(y_true[self.value], y_pred[self.value])

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Ensure the id columns are of string type and sort by them.
        id_column_sub = submission.columns[0]
        id_column_true = ground_truth.columns[0]
        submission[id_column_sub] = submission[id_column_sub].astype(str)
        ground_truth[id_column_true] = ground_truth[id_column_true].astype(str)
        submission = submission.sort_values(by=id_column_sub).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=id_column_true).reset_index(drop=True)

        if (submission[id_column_sub].values != ground_truth[id_column_true].values).any():
            raise InvalidSubmissionError("The id column values do not match between submission and ground truth. Please ensure they are identical and aligned.")

        sub_cols = set(submission.columns)
        true_cols = set(ground_truth.columns)

        missing_cols = true_cols - sub_cols
        extra_cols = sub_cols - true_cols
            
        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."