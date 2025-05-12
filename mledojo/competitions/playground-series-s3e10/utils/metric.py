from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError
from sklearn.metrics import log_loss

class PlaygroundSeriesS3e10Metrics(CompetitionMetrics):
    """Metric class for Playground Series S3E10 competition using Log Loss"""
    def __init__(self, value: str = "Class", higher_is_better: bool = False):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure the 'id' column is of string type and sort by it
        id_col_true = y_true.columns[0]
        id_col_pred = y_pred.columns[0]
        y_true[id_col_true] = y_true[id_col_true].astype(str)
        y_pred[id_col_pred] = y_pred[id_col_pred].astype(str)
        y_true = y_true.sort_values(by=id_col_true).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=id_col_pred).reset_index(drop=True)
        
        # Get the true labels and predicted probabilities
        y_true_labels = y_true[self.value].values
        y_pred_probs = y_pred[self.value].values
        
        # Bound predictions away from 0 and 1
        eps = 1e-15
        y_pred_probs = np.clip(y_pred_probs, eps, 1 - eps)
        
        # Calculate log loss
        score = - np.mean(y_true_labels * np.log(y_pred_probs) + (1 - y_true_labels) * np.log(1 - y_pred_probs))
        return score

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Convert the id columns to string type and sort by them
        submission_id_col = submission.columns[0]
        ground_truth_id_col = ground_truth.columns[0]
        submission[submission_id_col] = submission[submission_id_col].astype(str)
        ground_truth[ground_truth_id_col] = ground_truth[ground_truth_id_col].astype(str)
        submission = submission.sort_values(by=submission_id_col).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=ground_truth_id_col).reset_index(drop=True)

        # Check if id columns are identical
        if not np.array_equal(submission[submission_id_col].values, ground_truth[ground_truth_id_col].values):
            raise InvalidSubmissionError("The id values do not match between submission and ground truth. Please ensure the first column values are identical and in the same order.")

        sub_cols = set(submission.columns)
        true_cols = set(ground_truth.columns)

        missing_cols = true_cols - sub_cols
        extra_cols = sub_cols - true_cols

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."