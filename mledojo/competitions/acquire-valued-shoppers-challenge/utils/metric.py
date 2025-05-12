from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError
from sklearn.metrics import roc_auc_score

class AcquireValuedShoppersChallengeMetrics(CompetitionMetrics):
    def __init__(self, value: str = "repeatProbability", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure the id column is of string type
        y_true[y_true.columns[0]] = y_true[y_true.columns[0]].astype(str)
        y_pred[y_pred.columns[0]] = y_pred[y_pred.columns[0]].astype(str)
        # Sort both dataframes by the id column (first column)
        y_true_sorted = y_true.sort_values(by=y_true.columns[0]).reset_index(drop=True)
        y_pred_sorted = y_pred.sort_values(by=y_pred.columns[0]).reset_index(drop=True)
        # Calculate and return the ROC AUC score
        return roc_auc_score(y_true_sorted[self.value], y_pred_sorted[self.value])

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(
                f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows."
            )

        submission[submission.columns[0]] = submission[submission.columns[0]].astype(str)
        ground_truth[ground_truth.columns[0]] = ground_truth[ground_truth.columns[0]].astype(str)
        submission = submission.sort_values(by=submission.columns[0]).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=ground_truth.columns[0]).reset_index(drop=True)

        if not (submission[submission.columns[0]].values == ground_truth[ground_truth.columns[0]].values).all():
            raise InvalidSubmissionError("The id values in the submission do not match those in the ground truth. Please ensure they are identical and in the same order.")

        required_columns = set(ground_truth.columns)
        submission_columns = set(submission.columns)
        missing_cols = required_columns - submission_columns
        extra_cols = submission_columns - required_columns

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."