from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError
from sklearn.metrics import roc_auc_score

class TabularPlaygroundSeriesMar2021Metrics(CompetitionMetrics):
    def __init__(self, value: str = "target", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        if "id" not in y_true.columns or "id" not in y_pred.columns:
            raise InvalidSubmissionError("Both y_true and y_pred must contain an 'id' column.")
        y_true["id"] = y_true["id"].astype(str)
        y_pred["id"] = y_pred["id"].astype(str)
        y_true = y_true.sort_values("id").reset_index(drop=True)
        y_pred = y_pred.sort_values("id").reset_index(drop=True)
        return roc_auc_score(y_true[self.value], y_pred[self.value])

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        required_columns = {"id", "target"}
        submission_columns = set(submission.columns)
        ground_truth_columns = set(ground_truth.columns)

        missing_in_submission = required_columns - submission_columns
        if missing_in_submission:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_in_submission)}.")

        for col in ["id"]:
            submission[col] = submission[col].astype(str)
            ground_truth[col] = ground_truth[col].astype(str)

        submission = submission.sort_values("id").reset_index(drop=True)
        ground_truth = ground_truth.sort_values("id").reset_index(drop=True)

        if not all(submission["id"].values == ground_truth["id"].values):
            raise InvalidSubmissionError("The 'id' column values do not match between submission and ground truth. Please ensure they are identical.")

        extra_cols = submission_columns - ground_truth_columns
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."