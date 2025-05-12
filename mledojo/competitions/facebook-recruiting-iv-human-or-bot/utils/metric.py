from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError
from sklearn.metrics import roc_auc_score

class FacebookRecruitingIvHumanOrBotMetrics(CompetitionMetrics):
    def __init__(self, value: str = "prediction", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure bidder_id columns are strings and sort both DataFrames by bidder_id (the first column)
        y_true[y_true.columns[0]] = y_true[y_true.columns[0]].astype(str)
        y_pred[y_pred.columns[0]] = y_pred[y_pred.columns[0]].astype(str)
        y_true = y_true.sort_values(by=y_true.columns[0]).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=y_pred.columns[0]).reset_index(drop=True)
        
        # Compute ROC AUC score using the ground truth 'outcome' column and submission predictions
        return roc_auc_score(y_true[self.value], y_pred[self.value])

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        # Ensure submission has the required columns: bidder_id and prediction
        required_sub_cols = {"bidder_id", "prediction"}
        missing_sub = required_sub_cols - set(submission.columns)
        if missing_sub:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_sub)}.")

        # Ensure ground truth has the required columns: bidder_id and outcome
        required_gt_cols = {"bidder_id", "prediction"}
        missing_gt = required_gt_cols - set(ground_truth.columns)
        if missing_gt:
            raise InvalidSubmissionError(f"Missing required columns in ground truth: {', '.join(missing_gt)}.")

        # Convert bidder_id columns to string type
        submission["bidder_id"] = submission["bidder_id"].astype(str)
        ground_truth["bidder_id"] = ground_truth["bidder_id"].astype(str)

        # Sort both DataFrames by bidder_id
        submission = submission.sort_values(by="bidder_id").reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by="bidder_id").reset_index(drop=True)

        # Check if bidder_id values match between submission and ground truth
        if not (submission["bidder_id"].values == ground_truth["bidder_id"].values).all():
            raise InvalidSubmissionError("Bidder_id values do not match between submission and ground truth. Please ensure the first column values are identical.")

        return "Submission is valid."