from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError
from sklearn.metrics import f1_score

class NlpGettingStartedMetrics(CompetitionMetrics):
    def __init__(self, value: str = "target", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Convert id columns (first column) to string type for both DataFrames
        id_col_true = y_true.columns[0]
        id_col_pred = y_pred.columns[0]
        y_true[id_col_true] = y_true[id_col_true].astype(str)
        y_pred[id_col_pred] = y_pred[id_col_pred].astype(str)
        # Sort both DataFrames by id column
        y_true = y_true.sort_values(by=id_col_true).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=id_col_pred).reset_index(drop=True)
        # Calculate F1 score using the specified "target" column
        score = f1_score(y_true[self.value], y_pred[self.value])
        return score

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

        # Convert id columns to string type
        id_col_sub = submission.columns[0]
        id_col_true = ground_truth.columns[0]
        submission[id_col_sub] = submission[id_col_sub].astype(str)
        ground_truth[id_col_true] = ground_truth[id_col_true].astype(str)
        # Sort submission and ground truth by id column
        submission_sorted = submission.sort_values(by=id_col_sub)
        ground_truth_sorted = ground_truth.sort_values(by=id_col_true)

        # Check if id columns match exactly
        if (submission_sorted[id_col_sub].values != ground_truth_sorted[id_col_true].values).any():
            raise InvalidSubmissionError(
                "ID column values do not match between submission and ground truth. Please ensure the IDs are identical."
            )

        sub_cols = set(submission.columns)
        true_cols = set(ground_truth.columns)

        missing_cols = true_cols - sub_cols
        extra_cols = sub_cols - true_cols

        if missing_cols:
            raise InvalidSubmissionError(
                f"Missing required columns in submission: {', '.join(missing_cols)}."
            )
        if extra_cols:
            raise InvalidSubmissionError(
                f"Extra unexpected columns found in submission: {', '.join(extra_cols)}."
            )

        return "Submission is valid."