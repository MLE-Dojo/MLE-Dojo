from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError
from sklearn.metrics import log_loss

class KobeBryantShotSelectionMetrics(CompetitionMetrics):
    """Metric class for Kobe Bryant Shot Selection competition using Log Loss."""
    def __init__(self, value: str = "shot_made_flag", higher_is_better: bool = False):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure the 'shot_id' column is of string type so that sorting is consistent.
        if "shot_id" not in y_true.columns or "shot_id" not in y_pred.columns:
            raise InvalidSubmissionError("Both ground truth and submission must contain the 'shot_id' column.")
        
        y_true["shot_id"] = y_true["shot_id"].astype(str)
        y_pred["shot_id"] = y_pred["shot_id"].astype(str)
        y_true = y_true.sort_values(by="shot_id").reset_index(drop=True)
        y_pred = y_pred.sort_values(by="shot_id").reset_index(drop=True)
        
        # Calculate log loss between the predictions and the true labels.
        score = log_loss(y_true[self.value], y_pred[self.value])
        return score

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Check for required 'shot_id' column.
        if "shot_id" not in submission.columns or "shot_id" not in ground_truth.columns:
            raise InvalidSubmissionError("Both submission and ground truth must contain the 'shot_id' column.")

        # Convert 'shot_id' column to string type
        submission["shot_id"] = submission["shot_id"].astype(str)
        ground_truth["shot_id"] = ground_truth["shot_id"].astype(str)
        # Sort the submission and ground truth by 'shot_id'
        submission = submission.sort_values(by="shot_id").reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by="shot_id").reset_index(drop=True)

        # Check if shot_id columns match
        if not (submission["shot_id"].values == ground_truth["shot_id"].values).all():
            raise InvalidSubmissionError("The 'shot_id' values in submission do not match those in the ground truth. Please ensure they are identical and in the same order.")

        # Check for required columns
        required_columns = {"shot_id", self.value}
        sub_cols = set(submission.columns)
        true_cols = set(ground_truth.columns)

        missing_cols = required_columns - sub_cols
        extra_cols = sub_cols - true_cols
            
        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."