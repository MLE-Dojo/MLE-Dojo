from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class CrowdflowerWeatherTwitterMetrics(CompetitionMetrics):
    """Metric class for the crowdflower-weather-twitter competition using RMSE for evaluation."""
    def __init__(self, value: str = "rmse", higher_is_better: bool = False):
        # Note: In this competition, lower RMSE is better so higher_is_better is False.
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Convert the id column (first column) to string type for both dataframes
        id_col = y_true.columns[0]
        y_true[id_col] = y_true[id_col].astype(str)
        y_pred[y_pred.columns[0]] = y_pred[y_pred.columns[0]].astype(str)
        
        # Sort both dataframes by the id column and reset index
        y_true = y_true.sort_values(by=id_col).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=y_pred.columns[0]).reset_index(drop=True)
        
        # Get the list of label columns (all columns except the first id column)
        label_cols = list(y_true.columns[1:])
        
        # Check that the label columns exist in the prediction dataframe as well.
        if set(label_cols) != set(y_pred.columns[1:]):
            raise InvalidSubmissionError("Prediction columns do not match ground truth columns.")

        # Calculate the squared differences on all label columns
        diff = y_true[label_cols].values - y_pred[label_cols].values
        mse = np.mean(diff ** 2)
        rmse = np.sqrt(mse)
        return rmse

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Convert the id column in both dataframes to string type
        id_col_sub = submission.columns[0]
        id_col_true = ground_truth.columns[0]
        submission[id_col_sub] = submission[id_col_sub].astype(str)
        ground_truth[id_col_true] = ground_truth[id_col_true].astype(str)
        
        # Sort the dataframes by the id column
        submission = submission.sort_values(by=id_col_sub).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=id_col_true).reset_index(drop=True)

        # Check if the id columns match exactly
        if not np.array_equal(submission[id_col_sub].values, ground_truth[id_col_true].values):
            raise InvalidSubmissionError("The id column values do not match between submission and ground truth. Please ensure the first column values are identical and in the same order.")

        submission_cols = set(submission.columns)
        ground_truth_cols = set(ground_truth.columns)
        
        missing_cols = ground_truth_cols - submission_cols
        extra_cols = submission_cols - ground_truth_cols
            
        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."