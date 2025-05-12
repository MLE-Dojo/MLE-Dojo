from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class IcecubeNeutrinosInDeepIceMetrics(CompetitionMetrics):
    """
    Metric class for the IceCube Neutrinos in Deep Ice competition.
    Evaluates submissions based on the mean angular error between
    the predicted and true neutrino directions.
    """
    def __init__(self, value: str = "azimuth", higher_is_better: bool = False):
        # Although both 'azimuth' and 'zenith' are used, we set value to one of the targets.
        # The evaluation function uses both columns.
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure event_id is string for proper alignment
        y_true["event_id"] = y_true["event_id"].astype(str)
        y_pred["event_id"] = y_pred["event_id"].astype(str)
        
        # Sort both dataframes by 'event_id'
        y_true = y_true.sort_values(by="event_id").reset_index(drop=True)
        y_pred = y_pred.sort_values(by="event_id").reset_index(drop=True)

        # Extract the true and predicted angles
        true_azimuth = y_true["azimuth"].to_numpy()
        true_zenith  = y_true["zenith"].to_numpy()
        pred_azimuth = y_pred["azimuth"].to_numpy()
        pred_zenith  = y_pred["zenith"].to_numpy()
        
        # Compute cosine of the angular error using spherical law of cosines
        # cos(error) = sin(true_zenith)*sin(pred_zenith)*cos(true_azimuth - pred_azimuth)
        #              + cos(true_zenith)*cos(pred_zenith)
        cos_error = (np.sin(true_zenith) * np.sin(pred_zenith) * 
                     np.cos(true_azimuth - pred_azimuth) +
                     np.cos(true_zenith) * np.cos(pred_zenith))
        # Clip values to avoid numerical errors outside the domain of arccos
        cos_error = np.clip(cos_error, -1, 1)
        angular_errors = np.arccos(cos_error)
        
        # Return the mean angular error as the score (lower is better)
        return float(np.mean(angular_errors))

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        # Check types for submission and ground_truth
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        # Required columns for the competition
        required_columns = {"event_id", "azimuth", "zenith"}
        
        sub_cols = set(submission.columns)
        true_cols = set(ground_truth.columns)
        
        missing_cols = required_columns - sub_cols
        extra_cols = sub_cols - required_columns
        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        # Extra columns are allowed provided they don't cause misalignment, but here we flag them
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Convert event_id to string type to ensure alignment
        submission["event_id"] = submission["event_id"].astype(str)
        ground_truth["event_id"] = ground_truth["event_id"].astype(str)
        
        # Sort the submission and ground truth by the 'event_id' column
        submission = submission.sort_values(by="event_id").reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by="event_id").reset_index(drop=True)

        # Check if event_id columns match between submission and ground truth
        if not np.array_equal(submission["event_id"].values, ground_truth["event_id"].values):
            raise InvalidSubmissionError("The event_id values do not match between submission and ground truth. Please ensure the event_ids are identical.")

        return "Submission is valid."