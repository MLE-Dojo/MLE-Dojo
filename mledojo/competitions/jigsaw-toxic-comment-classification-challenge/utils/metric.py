from typing import Any
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError


class JigsawToxicCommentClassificationChallengeMetrics(CompetitionMetrics):
    """Metric class for Jigsaw Toxic Comment Classification Challenge using mean column-wise ROC AUC"""

    def __init__(self, value: str = None, higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Sort both dataframes by id index
        y_true = y_true.set_index("id").sort_index()
        y_pred = y_pred.set_index("id").sort_index()
        columns = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        keep_mask = y_true[columns].sum(axis=1) >= 0

        y_true_filtered = y_true[keep_mask]
        y_pred_filtered = y_pred[keep_mask]

        y_true_array = y_true_filtered.to_numpy()
        y_pred_array = y_pred_filtered.to_numpy()

        return roc_auc_score(y_true_array, y_pred_array, average="macro")

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

        # Check if first columns are identical
        if (submission[submission.columns[0]].values != (ground_truth[ground_truth.columns[0]].values)).any():
            raise InvalidSubmissionError(
                "First column values do not match between submission and ground truth. Please ensure the first column values are identical."
            )

        sub_cols = set(submission.columns)
        true_cols = set(ground_truth.columns)

        missing_cols = true_cols - sub_cols
        extra_cols = sub_cols - true_cols

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."
