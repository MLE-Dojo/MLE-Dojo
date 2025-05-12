from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class PetfinderPawpularityScoreMetrics(CompetitionMetrics):
    """Metric class for Petfinder-Pawpularity-Score competition using Root Mean Squared Error (RMSE)"""
    def __init__(self, value: str = "Pawpularity", higher_is_better: bool = False):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Convert the identifier column to string type
        y_true["Id"] = y_true["Id"].astype(str)
        y_pred["Id"] = y_pred["Id"].astype(str)
        # Sort both DataFrames by the identifier column
        y_true = y_true.sort_values(by="Id").reset_index(drop=True)
        y_pred = y_pred.sort_values(by="Id").reset_index(drop=True)
        # Compute RMSE
        differences = y_true[self.value] - y_pred[self.value]
        mse = np.mean(differences ** 2)
        rmse = np.sqrt(mse)
        return float(rmse)

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(
                f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows."
            )

        # Convert the identifier column to string type
        submission["Id"] = submission["Id"].astype(str)
        ground_truth["Id"] = ground_truth["Id"].astype(str)
        # Sort both DataFrames by the identifier column
        submission = submission.sort_values(by="Id").reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by="Id").reset_index(drop=True)

        # Check if the identifier columns are identical
        if not (submission["Id"].values == ground_truth["Id"].values).all():
            raise InvalidSubmissionError("The 'Id' column values do not match between submission and ground truth. Please ensure they are identical and correctly ordered.")

        # Check that submission has exactly the required columns: Id and Pawpularity
        required_columns = {"Id", "Pawpularity"}
        submission_cols = set(submission.columns)
        
        if submission_cols != required_columns:
            missing_cols = required_columns - submission_cols
            extra_cols = submission_cols - required_columns
            
            if missing_cols:
                raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
            if extra_cols:
                raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}. Submission should only contain 'Id' and 'Pawpularity' columns.")

        return "Submission is valid."