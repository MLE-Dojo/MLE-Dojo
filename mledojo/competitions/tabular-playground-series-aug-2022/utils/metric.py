from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError
from sklearn.metrics import roc_auc_score

class TabularPlaygroundSeriesAug2022Metrics(CompetitionMetrics):
    """Metric class for Tabular Playground Series Aug 2022 competition using ROC AUC."""
    def __init__(self, value: str = "failure", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure the id column is of string type
        y_true[y_true.columns[0]] = y_true[y_true.columns[0]].astype(str)
        y_pred[y_pred.columns[0]] = y_pred[y_pred.columns[0]].astype(str)
        # Sort both dataframes by the id column (the first column)
        y_true = y_true.sort_values(by=y_true.columns[0]).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=y_pred.columns[0]).reset_index(drop=True)
        # Return ROC AUC score using the 'failure' column
        return roc_auc_score(y_true[self.value], y_pred[self.value])

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Ensure the id column is of string type for both submission and ground truth
        submission[submission.columns[0]] = submission[submission.columns[0]].astype(str)
        ground_truth[ground_truth.columns[0]] = ground_truth[ground_truth.columns[0]].astype(str)
        # Sort the submission and ground truth by the id column (the first column)
        submission = submission.sort_values(by=submission.columns[0])
        ground_truth = ground_truth.sort_values(by=ground_truth.columns[0])

        # Check if id values are identical between submission and ground truth
        if not np.array_equal(submission[submission.columns[0]].values, ground_truth[ground_truth.columns[0]].values):
            raise InvalidSubmissionError("The id column values do not match between submission and ground truth. Please ensure they are identical.")

        submission_cols = set(submission.columns)
        ground_truth_cols = set(ground_truth.columns)
        
        missing_cols = ground_truth_cols - submission_cols
        extra_cols = submission_cols - ground_truth_cols

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."