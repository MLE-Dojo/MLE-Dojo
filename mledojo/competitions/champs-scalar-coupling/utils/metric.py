from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class ChampsScalarCouplingMetrics(CompetitionMetrics):
    """
    Metric class for CHAMPS Scalar Coupling competition.
    
    The score is computed as the average over scalar coupling types of:
        log( max(MAE, 1e-9) )
    where, for each coupling type t:
        MAE_t = (1/n_t) * sum(|y_true - y_pred|)
    Lower scores are better. For perfect predictions, the score is approximately -20.7232.
    
    The submission file should contain two columns:
        id, scalar_coupling_constant
    """
    def __init__(self, value: str = "scalar_coupling_constant", higher_is_better: bool = False):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure the id column is of string type
        y_true['id'] = y_true['id'].astype(str)
        y_pred['id'] = y_pred['id'].astype(str)
        
        # Sort both DataFrames by the 'id' column
        y_true = y_true.sort_values(by='id').reset_index(drop=True)
        y_pred = y_pred.sort_values(by='id').reset_index(drop=True)
        
        # Merge on the id column. 
        # Expect y_true to contain at least: id, scalar_coupling_constant, and coupling type (assumed column name: "type")
        merged = pd.merge(y_true, y_pred, on="id", how="inner", suffixes=("_true", "_pred"))
        if merged.shape[0] != y_true.shape[0]:
            raise InvalidSubmissionError("Mismatch between submission and ground truth ids after merge. Please ensure that all ids are present.")

        # Check that the required coupling type column exists
        if "type" not in merged.columns:
            raise InvalidSubmissionError("Ground truth must contain a 'type' column for scalar coupling type.")

        # Group by scalar coupling type and compute log(MAE) with a floor of 1e-9
        scores = []
        for t, group in merged.groupby("type"):
            # Compute mean absolute error for this group
            mae = np.mean(np.abs(group[f"{self.value}_true"] - group[f"{self.value}_pred"]))
            mae = max(mae, 1e-9)
            scores.append(np.log(mae))
        # Average the log(MAE) over all groups
        score = np.mean(scores)
        return score

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        # Check that number of rows matches (based on submission ids)
        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Ensure the 'id' column exists in both DataFrames
        if 'id' not in submission.columns:
            raise InvalidSubmissionError("Submission must contain 'id' column.")
        if 'id' not in ground_truth.columns:
            raise InvalidSubmissionError("Ground truth must contain 'id' column.")

        # Convert 'id' columns to string type
        submission['id'] = submission['id'].astype(str)
        ground_truth['id'] = ground_truth['id'].astype(str)

        # Sort both DataFrames by the 'id' column
        submission_sorted = submission.sort_values(by='id').reset_index(drop=True)
        ground_truth_sorted = ground_truth.sort_values(by='id').reset_index(drop=True)

        # Check if the 'id' columns match exactly
        if not np.array_equal(submission_sorted['id'].values, ground_truth_sorted['id'].values):
            raise InvalidSubmissionError("The 'id' column values do not match between submission and ground truth. Please ensure they are identical and in the same order.")

        # Validate that submission has exactly the required columns: 'id' and the target value (scalar_coupling_constant)
        required_cols = {"id", self.value}
        sub_cols = set(submission.columns)
        missing_cols = required_cols - sub_cols
        extra_cols = sub_cols - required_cols

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."