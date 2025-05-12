from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError
from sklearn.metrics import log_loss

class LmsysChatbotArenaMetrics(CompetitionMetrics):
    """
    Metric class for LMSYS-CHATBOT-ARENA competition using multi-class log loss.
    """
    def __init__(self, value: str = "winner_model_a", higher_is_better: bool = False):
        super().__init__(higher_is_better)
        self.value = value  # The primary column (one of the target columns)
        # Define the target columns expected in both ground truth and submission
        self.target_cols = ["winner_model_a", "winner_model_b", "winner_tie"]

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Convert the id column to string
        y_true[y_true.columns[0]] = y_true[y_true.columns[0]].astype(str)
        y_pred[y_pred.columns[0]] = y_pred[y_pred.columns[0]].astype(str)
        
        # Sort both dataframes by the 'id' column (assumed to be the first column)
        y_true = y_true.sort_values(by=y_true.columns[0]).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=y_pred.columns[0]).reset_index(drop=True)
        
        # Extract the target probability columns for evaluation
        try:
            y_true_targets = y_true[self.target_cols].values
            y_pred_targets = y_pred[self.target_cols].values
        except KeyError as e:
            raise InvalidSubmissionError(f"Submission or ground truth is missing one of the required target columns: {e}")
        
        # Calculate multi-class log loss
        # Remove the eps parameter as it's causing an error
        score = log_loss(y_true_targets, y_pred_targets)
        return score

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        
        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")
        
        # Convert the id column to string type for both dataframes
        submission[submission.columns[0]] = submission[submission.columns[0]].astype(str)
        ground_truth[ground_truth.columns[0]] = ground_truth[ground_truth.columns[0]].astype(str)
        
        # Sort both dataframes by the id column
        submission = submission.sort_values(by=submission.columns[0])
        ground_truth = ground_truth.sort_values(by=ground_truth.columns[0])
        
        # Check if the id columns are identical
        if (submission[submission.columns[0]].values != ground_truth[ground_truth.columns[0]].values).any():
            raise InvalidSubmissionError("The id column values do not match between submission and ground truth. Please ensure they are identical and in the same order.")
        
        # Define the required columns: id column plus target columns
        required_cols = set([submission.columns[0]] + self.target_cols)
        submission_cols = set(submission.columns)
        
        missing_cols = required_cols - submission_cols
        extra_cols = submission_cols - required_cols
        
        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")
        
        return "Submission is valid."