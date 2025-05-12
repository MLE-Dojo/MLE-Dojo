from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError
from sklearn.metrics import log_loss

class StatoilIcebergClassifierChallengeMetrics(CompetitionMetrics):
    """Metric class for Statoil Iceberg Classifier Challenge using Log Loss"""
    def __init__(self, value: str = "is_iceberg", higher_is_better: bool = False):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure the id columns are of type string and sort by the id column.
        id_col_true = y_true.columns[0]
        id_col_pred = y_pred.columns[0]
        y_true[id_col_true] = y_true[id_col_true].astype(str)
        y_pred[id_col_pred] = y_pred[id_col_pred].astype(str)
        y_true = y_true.sort_values(by=id_col_true).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=id_col_pred).reset_index(drop=True)
        
        # Compute log loss using sklearn
        score = log_loss(y_true[self.value], y_pred[self.value])
        return score

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Convert id columns to string type
        id_col_sub = submission.columns[0]
        id_col_gt = ground_truth.columns[0]
        submission[id_col_sub] = submission[id_col_sub].astype(str)
        ground_truth[id_col_gt] = ground_truth[id_col_gt].astype(str)

        # Sort both submission and ground truth by their id columns
        submission = submission.sort_values(by=id_col_sub).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=id_col_gt).reset_index(drop=True)

        # Check if id columns are identical
        if not (submission[id_col_sub].values == ground_truth[id_col_gt].values).all():
            raise InvalidSubmissionError("The id values in the submission do not match those in the ground truth. Please ensure they are identical and in the same order.")

        # Check for required columns: expecting id and is_iceberg
        required_columns = {"id", "is_iceberg"}
        submission_columns = set(submission.columns)
        ground_truth_columns = set(ground_truth.columns)
        
        missing_cols = required_columns - submission_columns
        extra_cols = submission_columns - required_columns
        
        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        # Optionally, if extra columns are not allowed, uncomment the following lines:
        # if extra_cols:
        #     raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")
        
        # Check that predicted probabilities are between 0 and 1
        if submission["is_iceberg"].min() < 0 or submission["is_iceberg"].max() > 1:
            raise InvalidSubmissionError("Predicted probabilities in 'is_iceberg' must be between 0 and 1.")
            
        return "Submission is valid."