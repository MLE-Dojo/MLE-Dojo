from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError
from sklearn.metrics import log_loss

class BnpParibasCardifClaimsManagementMetrics(CompetitionMetrics):
    """
    Metric class for BNP Paribas Cardif Claims Management competition using Log Loss.
    Lower log loss is better.
    """
    def __init__(self, value: str = "PredictedProb", higher_is_better: bool = False):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure the ID columns are of type str
        y_true["ID"] = y_true["ID"].astype(str)
        y_pred["ID"] = y_pred["ID"].astype(str)
        # Sort both dataframes by the "ID" column
        y_true = y_true.sort_values(by="ID").reset_index(drop=True)
        y_pred = y_pred.sort_values(by="ID").reset_index(drop=True)
        

        true_labels = y_true[self.value].values
        preds = y_pred[self.value].values
        
        # Bound predicted probabilities to avoid log(0)
        preds = np.clip(preds, 1e-15, 1 - 1e-15)
        
        # Compute log loss. lower is better.
        score = log_loss(true_labels, preds)
        return score

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        # Check for pandas DataFrame input
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        
        # Expected columns: submission should have 'ID' and 'PredictedProb'
        required_submission_cols = {"ID", "PredictedProb"}
        if set(submission.columns) != required_submission_cols:
            missing_cols = required_submission_cols - set(submission.columns)
            extra_cols = set(submission.columns) - required_submission_cols
            if missing_cols:
                raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
            if extra_cols:
                raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")
        
        # Ground truth is expected to have 'ID' and 'PredictedProb'
        required_gt_cols = {"ID", "PredictedProb"}
        if set(ground_truth.columns) != required_gt_cols:
            missing_cols = required_gt_cols - set(ground_truth.columns)
            extra_cols = set(ground_truth.columns) - required_gt_cols
            if missing_cols:
                raise InvalidSubmissionError(f"Missing required columns in ground truth: {', '.join(missing_cols)}.")
            if extra_cols:
                raise InvalidSubmissionError(f"Extra unexpected columns found in ground truth: {', '.join(extra_cols)}.")
        
        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(
                f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)})."
            )
        
        # Convert "ID" column in both DataFrames to string and sort by "ID"
        submission["ID"] = submission["ID"].astype(str)
        ground_truth["ID"] = ground_truth["ID"].astype(str)
        submission = submission.sort_values(by="ID").reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by="ID").reset_index(drop=True)
        
        # Check if the "ID" columns are identical
        if not (submission["ID"].values == ground_truth["ID"].values).all():
            raise InvalidSubmissionError("The 'ID' values in submission do not match those in ground truth. Please ensure they are identical.")
        
        return "Submission is valid."