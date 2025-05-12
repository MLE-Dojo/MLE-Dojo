from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError
from sklearn.metrics import mean_absolute_error

class PredictVolcanicEruptionsIngvOeMetrics(CompetitionMetrics):
    """Metric class for Predict Volcanic Eruptions INGV OE competition using Mean Absolute Error (MAE)"""
    def __init__(self, value: str = "time_to_eruption", higher_is_better: bool = False):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure the id column ("segment_id") is of type string
        y_true['segment_id'] = y_true['segment_id'].astype(str)
        y_pred['segment_id'] = y_pred['segment_id'].astype(str)
        
        # Sort both dataframes by "segment_id"
        y_true = y_true.sort_values(by='segment_id').reset_index(drop=True)
        y_pred = y_pred.sort_values(by='segment_id').reset_index(drop=True)
        
        # Calculate and return the mean absolute error for the "time_to_eruption" column
        return mean_absolute_error(y_true[self.value], y_pred[self.value])

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError(
                "Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame."
            )
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError(
                "Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame."
            )
        
        required_columns = {"segment_id", "time_to_eruption"}
        if set(ground_truth.columns) != required_columns:
            raise InvalidSubmissionError(
                f"Ground truth must have columns: {', '.join(required_columns)}."
            )
        if set(submission.columns) != required_columns:
            raise InvalidSubmissionError(
                f"Submission must have columns: {', '.join(required_columns)}."
            )
            
        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(
                f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows."
            )
        
        # Ensure the "segment_id" column is of string type and sort both dataframes by "segment_id"
        submission['segment_id'] = submission['segment_id'].astype(str)
        ground_truth['segment_id'] = ground_truth['segment_id'].astype(str)
        
        submission = submission.sort_values(by="segment_id").reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by="segment_id").reset_index(drop=True)
        
        if not (submission['segment_id'].values == ground_truth['segment_id'].values).all():
            raise InvalidSubmissionError(
                "The segment_id values do not match between submission and ground truth. Please ensure they are identical and in the same order."
            )
        
        return "Submission is valid."