from typing import Any
import pandas as pd
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError
from sklearn.metrics import r2_score

class PlaygroundSeriesS4e5Metrics(CompetitionMetrics):
    """
    Metric class for playground-series-s4e5 competition using R2 Score.
    The submission file should contain the columns: id, FloodProbability.
    """
    def __init__(self, value: str = "FloodProbability", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure the id column (first column) is of string type for both dataframes
        y_true[y_true.columns[0]] = y_true[y_true.columns[0]].astype(str)
        y_pred[y_pred.columns[0]] = y_pred[y_pred.columns[0]].astype(str)
        
        # Sort both dataframes by the id column to align rows correctly
        y_true = y_true.sort_values(by=y_true.columns[0]).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=y_pred.columns[0]).reset_index(drop=True)
        
        # Calculate and return the R2 score using the FloodProbability column
        return r2_score(y_true[self.value], y_pred[self.value])

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Convert the id column (first column) to string type
        submission[submission.columns[0]] = submission[submission.columns[0]].astype(str)
        ground_truth[ground_truth.columns[0]] = ground_truth[ground_truth.columns[0]].astype(str)
        
        # Sort both submission and ground truth by the id column
        submission = submission.sort_values(by=submission.columns[0]).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=ground_truth.columns[0]).reset_index(drop=True)

        # Check if the id columns match between submission and ground truth
        if (submission[submission.columns[0]].values != ground_truth[ground_truth.columns[0]].values).any():
            raise InvalidSubmissionError("The id values in the submission do not match the ground truth. Please ensure the first column values are identical.")

        # Check for required columns: id and FloodProbability
        required_columns = {ground_truth.columns[0], self.value}
        submission_columns = set(submission.columns)
        missing_columns = required_columns - submission_columns
        extra_columns = submission_columns - required_columns

        if missing_columns:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_columns)}.")
        if extra_columns:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_columns)}.")

        return "Submission is valid."