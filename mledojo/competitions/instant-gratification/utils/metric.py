from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError
from sklearn.metrics import roc_auc_score

class InstantGratificationMetrics(CompetitionMetrics):
    """Metric class for Instant Gratification competition using AUC-ROC"""
    def __init__(self, value: str = "target", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure that the id column is properly handled by converting it to string
        y_true[y_true.columns[0]] = y_true[y_true.columns[0]].astype(str)
        y_pred[y_pred.columns[0]] = y_pred[y_pred.columns[0]].astype(str)
        # Sort by id column which is the first column
        y_true_sorted = y_true.sort_values(by=y_true.columns[0]).reset_index(drop=True)
        y_pred_sorted = y_pred.sort_values(by=y_pred.columns[0]).reset_index(drop=True)
        # Calculate AUC using the "target" column
        return roc_auc_score(y_true_sorted[self.value], y_pred_sorted[self.value])
    
    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        
        required_columns = {ground_truth.columns[0], self.value}
        if set(submission.columns) != required_columns:
            missing = required_columns - set(submission.columns)
            extra = set(submission.columns) - required_columns
            if missing:
                raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing)}.")
            if extra:
                raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra)}.")
        
        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")
        
        # Convert the id columns to string
        submission[submission.columns[0]] = submission[submission.columns[0]].astype(str)
        ground_truth[ground_truth.columns[0]] = ground_truth[ground_truth.columns[0]].astype(str)
        # Sort by id column (first column)
        submission_sorted = submission.sort_values(by=submission.columns[0]).reset_index(drop=True)
        ground_truth_sorted = ground_truth.sort_values(by=ground_truth.columns[0]).reset_index(drop=True)
        
        # Check if id columns are identical
        if not np.array_equal(submission_sorted[submission_sorted.columns[0]].values, ground_truth_sorted[ground_truth_sorted.columns[0]].values):
            raise InvalidSubmissionError("The id column values do not match between submission and ground truth. Please ensure they are identical in both files.")
        
        return "Submission is valid."