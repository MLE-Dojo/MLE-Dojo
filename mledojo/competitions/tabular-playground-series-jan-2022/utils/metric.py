from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class TabularPlaygroundSeriesJan2022Metrics(CompetitionMetrics):
    def __init__(self, value: str = "num_sold", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Convert 'row_id' column to string type to ensure proper sorting
        y_true["row_id"] = y_true["row_id"].astype(str)
        y_pred["row_id"] = y_pred["row_id"].astype(str)
        # Sort both DataFrames by 'row_id'
        y_true = y_true.sort_values(by="row_id").reset_index(drop=True)
        y_pred = y_pred.sort_values(by="row_id").reset_index(drop=True)
        
        # Extract actual and predicted values for calculation
        actual = y_true[self.value].to_numpy()
        predicted = y_pred[self.value].to_numpy()
        
        # Avoid division by zero: if both actual and predicted are 0, define the error as 0.
        denominator = np.abs(actual) + np.abs(predicted)
        # Use np.where to handle the case where both values are zero
        errors = np.where((actual == 0) & (predicted == 0), 0, 200 * np.abs(actual - predicted) / denominator)
        
        # Return the mean SMAPE score
        return np.mean(errors)

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        
        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")
        
        # Convert 'row_id' columns to string type and sort by 'row_id'
        submission["row_id"] = submission["row_id"].astype(str)
        ground_truth["row_id"] = ground_truth["row_id"].astype(str)
        submission = submission.sort_values(by="row_id").reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by="row_id").reset_index(drop=True)
        
        # Check if 'row_id' values are identical between submission and ground truth
        if not np.array_equal(submission["row_id"].values, ground_truth["row_id"].values):
            raise InvalidSubmissionError("The 'row_id' column values do not match between submission and ground truth. Please ensure they are identical.")
        
        sub_cols = set(submission.columns)
        true_cols = set(ground_truth.columns)
        missing_cols = true_cols - sub_cols
        extra_cols = sub_cols - true_cols
        
        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")
        
        return "Submission is valid."