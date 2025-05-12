from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError
from sklearn.metrics import mean_squared_error

class FacialKeypointsDetectionMetrics(CompetitionMetrics):
    """Metric class for Facial Keypoints Detection competition using Root Mean Squared Error (RMSE)"""

    def __init__(self, value: str = "Location", higher_is_better: bool = False):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Convert 'RowId' column to string for consistent sorting
        y_true["RowId"] = y_true["RowId"].astype(str)
        y_pred["RowId"] = y_pred["RowId"].astype(str)
        
        # Sort both dataframes by 'RowId'
        y_true = y_true.sort_values(by="RowId").reset_index(drop=True)
        y_pred = y_pred.sort_values(by="RowId").reset_index(drop=True)
        
        # Compute the RMSE over the 'Location' column
        mse = mean_squared_error(y_true[self.value], y_pred[self.value])
        rmse = np.sqrt(mse)
        return rmse

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        required_columns = {"RowId", "ImageId", "FeatureName", "Location"}
        
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        if "RowId" not in submission.columns or "RowId" not in ground_truth.columns:
            raise InvalidSubmissionError("Both submission and ground truth must contain a 'RowId' column.")

        # Check for missing values in Location column
        if submission["Location"].isna().any():
            raise InvalidSubmissionError("Submission contains missing values in the 'Location' column. All Location values must be provided.")

        # Convert 'RowId' to string type and sort by it
        submission["RowId"] = submission["RowId"].astype(str)
        ground_truth["RowId"] = ground_truth["RowId"].astype(str)
        submission_sorted = submission.sort_values(by="RowId").reset_index(drop=True)
        ground_truth_sorted = ground_truth.sort_values(by="RowId").reset_index(drop=True)
        
        if not (submission_sorted["RowId"].values == ground_truth_sorted["RowId"].values).all():
            raise InvalidSubmissionError("RowId values do not match between submission and ground truth. Please ensure the 'RowId' values are identical and in the same order.")
        
        sub_cols = set(submission.columns)
        missing_cols = required_columns - sub_cols
        extra_cols = sub_cols - required_columns

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."