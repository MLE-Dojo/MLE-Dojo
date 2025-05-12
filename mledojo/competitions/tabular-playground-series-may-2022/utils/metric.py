from typing import Any
import pandas as pd
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError
from sklearn.metrics import roc_auc_score

class TabularPlaygroundSeriesMay2022Metrics(CompetitionMetrics):
    """Metric class for Tabular Playground Series May 2022 competition using ROC AUC"""
    def __init__(self, value: str = "target", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Convert the first column (id column) to string type
        id_col_true = y_true.columns[0]
        id_col_pred = y_pred.columns[0]
        y_true[id_col_true] = y_true[id_col_true].astype(str)
        y_pred[id_col_pred] = y_pred[id_col_pred].astype(str)

        # Sort both dataframes by the id column
        y_true = y_true.sort_values(by=id_col_true).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=id_col_pred).reset_index(drop=True)

        # Calculate and return the ROC AUC score
        return roc_auc_score(y_true[self.value], y_pred[self.value])

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Convert first column (id column) to string type
        id_col_sub = submission.columns[0]
        id_col_truth = ground_truth.columns[0]
        submission[id_col_sub] = submission[id_col_sub].astype(str)
        ground_truth[id_col_truth] = ground_truth[id_col_truth].astype(str)

        # Sort submission and ground truth by the id column
        submission = submission.sort_values(by=id_col_sub).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=id_col_truth).reset_index(drop=True)

        # Check if first column values (id) match between submission and ground truth
        if (submission[id_col_sub].values != ground_truth[id_col_truth].values).any():
            raise InvalidSubmissionError("Submission id column does not match ground truth id column. Please ensure the first column values are identical.")

        sub_cols = set(submission.columns)
        true_cols = set(ground_truth.columns)

        missing_cols = true_cols - sub_cols
        extra_cols = sub_cols - true_cols

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."