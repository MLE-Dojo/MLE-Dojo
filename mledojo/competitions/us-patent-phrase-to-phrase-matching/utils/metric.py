from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError
from scipy.stats import pearsonr

class UsPatentPhraseToPhraseMatchingMetrics(CompetitionMetrics):
    """Metric class for US Patent Phrase to Phrase Matching competition using Pearson correlation coefficient"""
    def __init__(self, value: str = "score", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure the first column (id) is of string type and sort by it
        y_true[y_true.columns[0]] = y_true[y_true.columns[0]].astype(str)
        y_pred[y_pred.columns[0]] = y_pred[y_pred.columns[0]].astype(str)
        y_true = y_true.sort_values(by=y_true.columns[0]).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=y_pred.columns[0]).reset_index(drop=True)
        
        # Calculate Pearson correlation coefficient between the true and predicted similarity scores
        correlation, _ = pearsonr(y_true[self.value], y_pred[self.value])
        return correlation

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(
                f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows."
            )

        # Ensure the first column (id) is of string type
        submission[submission.columns[0]] = submission[submission.columns[0]].astype(str)
        ground_truth[ground_truth.columns[0]] = ground_truth[ground_truth.columns[0]].astype(str)
        
        # Sort both submission and ground truth by the id column
        submission = submission.sort_values(by=submission.columns[0]).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=ground_truth.columns[0]).reset_index(drop=True)

        # Check if the id values of both dataframes match
        if not np.array_equal(submission[submission.columns[0]].values, ground_truth[ground_truth.columns[0]].values):
            raise InvalidSubmissionError("The id values in the submission do not match those in the ground truth. Please ensure the first column values are identical and in the correct order.")

        required_columns = set([ground_truth.columns[0], self.value])
        submission_columns = set(submission.columns)

        missing_cols = required_columns - submission_columns
        extra_cols = submission_columns - required_columns

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."