from typing import Any
import pandas as pd
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError
from sklearn.metrics import roc_auc_score

class PredictingRedHatBusinessValueMetrics(CompetitionMetrics):
    """Metric class for Predicting Red Hat Business Value competition using ROC AUC."""
    def __init__(self, value: str = "outcome", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Convert the id column (activity_id) to string type to ensure proper matching
        y_true[y_true.columns[0]] = y_true[y_true.columns[0]].astype(str)
        y_pred[y_pred.columns[0]] = y_pred[y_pred.columns[0]].astype(str)
        # Sort both DataFrames on the id column
        y_true = y_true.sort_values(by=y_true.columns[0]).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=y_pred.columns[0]).reset_index(drop=True)
        # Compute and return the ROC AUC score
        return roc_auc_score(y_true[self.value], y_pred[self.value])

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(
                f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows."
            )

        # Convert the id column (activity_id) to string type
        submission[submission.columns[0]] = submission[submission.columns[0]].astype(str)
        ground_truth[ground_truth.columns[0]] = ground_truth[ground_truth.columns[0]].astype(str)
        # Sort both DataFrames on the activity_id column
        submission = submission.sort_values(by=submission.columns[0]).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=ground_truth.columns[0]).reset_index(drop=True)

        # Check if the activity_id values are identical between submission and ground truth
        if (submission[submission.columns[0]].values != ground_truth[ground_truth.columns[0]].values).any():
            raise InvalidSubmissionError(
                "Activity IDs do not match between submission and ground truth. Please ensure the first column's values are identical and in the same order."
            )

        # Ensure both DataFrames contain the required columns: activity_id and outcome
        required_columns = {ground_truth.columns[0], self.value}
        submission_columns = set(submission.columns)

        missing_columns = required_columns - submission_columns
        extra_columns = submission_columns - required_columns

        if missing_columns:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_columns)}.")
        if extra_columns:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_columns)}.")

        return "Submission is valid."