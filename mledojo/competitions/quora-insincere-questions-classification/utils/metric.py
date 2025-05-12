from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError
from sklearn.metrics import f1_score

class QuoraInsincereQuestionsClassificationMetrics(CompetitionMetrics):
    """Metric class for Quora Insincere Questions Classification competition using F1 Score"""
    def __init__(self, value: str = "target", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Convert id column 'qid' to string type for both dataframes.
        y_true["qid"] = y_true["qid"].astype(str)
        y_pred["qid"] = y_pred["qid"].astype(str)
        # Sort both dataframes by the id column 'qid'
        y_true = y_true.sort_values(by="qid").reset_index(drop=True)
        y_pred = y_pred.sort_values(by="qid").reset_index(drop=True)
        # Calculate and return the F1 Score using the ground truth 'target' and submission 'prediction'
        return f1_score(y_true[self.value], y_pred["prediction"])

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(
                f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows."
            )

        # Ensure the id column 'qid' is present and convert it to string type
        if "qid" not in submission.columns:
            raise InvalidSubmissionError("Submission is missing the required 'qid' column.")
        if "qid" not in ground_truth.columns:
            raise InvalidSubmissionError("Ground truth is missing the required 'qid' column.")

        submission["qid"] = submission["qid"].astype(str)
        ground_truth["qid"] = ground_truth["qid"].astype(str)
        # Sort the submission and ground truth by the id column 'qid'
        submission = submission.sort_values(by="qid").reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by="qid").reset_index(drop=True)

        # Check if the 'qid' values match between submission and ground truth
        if not (submission["qid"].values == ground_truth["qid"].values).all():
            raise InvalidSubmissionError("The 'qid' values do not match between submission and ground truth. Please ensure they are identical and in the same order.")

        required_submission_cols = {"qid", "prediction"}
        required_ground_truth_cols = {"qid", self.value}

        sub_cols = set(submission.columns)
        true_cols = set(ground_truth.columns)

        missing_cols = required_submission_cols - sub_cols
        extra_cols = sub_cols - required_submission_cols

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        missing_true_cols = required_ground_truth_cols - true_cols
        if missing_true_cols:
            raise InvalidSubmissionError(f"Missing required columns in ground truth: {', '.join(missing_true_cols)}.")

        return "Submission is valid."