
from typing import Any
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class RanzcrClipCatheterLineClassificationMetrics(CompetitionMetrics):
    """Metric class for RanzcrClipCatheterLineClassification competition using AUC"""
    def __init__(self, value: str = None, higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Convert first column to string type
        y_true[y_true.columns[0]] = y_true[y_true.columns[0]].astype(str)
        y_pred[y_pred.columns[0]] = y_pred[y_pred.columns[0]].astype(str)
        labels = y_true.columns[1:]
        auc_scores = []
        for label in labels:
            try:
                auc = roc_auc_score(y_true[label], y_pred[label])
                auc_scores.append(auc)
            except ValueError:
                auc_scores.append(0.0)
        return np.mean(auc_scores)

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Convert first column to string type
        submission[submission.columns[0]] = submission[submission.columns[0]].astype(str)
        ground_truth[ground_truth.columns[0]] = ground_truth[ground_truth.columns[0]].astype(str)
        # Sort the submission and ground truth by the first column
        submission = submission.sort_values(by=submission.columns[0])
        ground_truth = ground_truth.sort_values(by=ground_truth.columns[0])

        # Check if first columns are identical
        if (submission[submission.columns[0]].values != (ground_truth[ground_truth.columns[0]].values)).any():
            raise InvalidSubmissionError("First column values do not match between submission and ground truth. Please ensure the first column values are identical.")

        sub_cols = set(submission.columns)
        true_cols = set(ground_truth.columns)

        missing_cols = true_cols - sub_cols
        extra_cols = sub_cols - true_cols
            
        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."