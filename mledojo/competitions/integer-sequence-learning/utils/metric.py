from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class IntegerSequenceLearningMetrics(CompetitionMetrics):
    """Metric class for Integer Sequence Learning competition using accuracy of predictions (percentage of correct next integer predictions)"""
    def __init__(self, value: str = "Last", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Convert the first column (ID column) to string type
        id_col = y_true.columns[0]
        y_true[id_col] = y_true[id_col].astype(str)
        y_pred[id_col] = y_pred[id_col].astype(str)

        # Sort both DataFrames by the ID column
        y_true = y_true.sort_values(by=id_col).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=id_col).reset_index(drop=True)

        # Calculate the accuracy: percentage of sequences with the correct predicted integer
        correct = (y_true[self.value] == y_pred[self.value]).sum()
        total = len(y_true)
        accuracy = correct / total
        return accuracy

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(
                f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows."
            )

        # Convert the first column (ID column) to string type and sort by it
        id_col_submission = submission.columns[0]
        id_col_ground_truth = ground_truth.columns[0]

        submission[id_col_submission] = submission[id_col_submission].astype(str)
        ground_truth[id_col_ground_truth] = ground_truth[id_col_ground_truth].astype(str)

        submission = submission.sort_values(by=id_col_submission)
        ground_truth = ground_truth.sort_values(by=id_col_ground_truth)

        # Check if the ID columns are identical
        if (submission[id_col_submission].values != ground_truth[id_col_ground_truth].values).any():
            raise InvalidSubmissionError("ID column values do not match between submission and ground truth. Please ensure the ID column values are identical.")

        sub_cols = set(submission.columns)
        true_cols = set(ground_truth.columns)

        missing_cols = true_cols - sub_cols
        extra_cols = sub_cols - true_cols

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."