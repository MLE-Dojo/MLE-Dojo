from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class BattlefinSBigDataCombineForecastingChallengeMetrics(CompetitionMetrics):
    """
    Metric class for BattleFin S Big Data Combine Forecasting Challenge using Mean Absolute Error (MAE).
    In this competition, submissions are evaluated by the mean absolute error between the predicted percentage change 
    and the actual percentage change. A lower MAE indicates better performance.
    """
    def __init__(self, value: str = "O1", higher_is_better: bool = False):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Convert first (ID) column to string type
        id_col = y_true.columns[0]
        y_true[id_col] = y_true[id_col].astype(str)
        y_pred[y_pred.columns[0]] = y_pred[y_pred.columns[0]].astype(str)
        
        # Sort by the ID column and reset index
        y_true = y_true.sort_values(by=id_col).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=y_pred.columns[0]).reset_index(drop=True)
        
        # Assume the first column is the ID and the remaining columns are outputs
        output_cols = list(y_true.columns[1:])
        
        # Calculate Mean Absolute Error across all output columns
        error = np.abs(y_true[output_cols].to_numpy() - y_pred[output_cols].to_numpy())
        mae = np.mean(error)
        return mae

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Convert the first column (ID column) to string type
        id_col = ground_truth.columns[0]
        submission[id_col] = submission[id_col].astype(str)
        ground_truth[id_col] = ground_truth[id_col].astype(str)

        # Sort the submission and ground truth by the ID column
        submission = submission.sort_values(by=id_col).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=id_col).reset_index(drop=True)

        # Check if first column values (IDs) are identical
        if not np.array_equal(submission[id_col].values, ground_truth[id_col].values):
            raise InvalidSubmissionError("First column values do not match between submission and ground truth. Please ensure the first column values are identical.")

        sub_cols = set(submission.columns)
        true_cols = set(ground_truth.columns)

        missing_cols = true_cols - sub_cols
        extra_cols = sub_cols - true_cols

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."