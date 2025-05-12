from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class ConwaysReverseGameOfLife2020Metrics(CompetitionMetrics):
    """
    Metric class for Conway's Reverse Game of Life 2020 competition.
    The score is computed as the mean absolute error (MAE) of the predicted starting boards compared
    to the ground truth starting boards, after stepping forward the given number of steps.
    Since the predictions are binary, the MAE is equivalent to 1 - classification accuracy.
    Lower scores are better.
    """
    def __init__(self, value: str = "start_0", higher_is_better: bool = False):
        # Note: Although 'value' is set to 'start_0' by default, the evaluation uses all columns starting with "start_".
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Convert first column (id) to string type
        y_true[y_true.columns[0]] = y_true[y_true.columns[0]].astype(str)
        y_pred[y_pred.columns[0]] = y_pred[y_pred.columns[0]].astype(str)
        # Sort by id column (the first column)
        y_true = y_true.sort_values(by=y_true.columns[0]).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=y_pred.columns[0]).reset_index(drop=True)
        
        # Identify all starting board columns (should be start_0 to start_624)
        cell_columns = [col for col in y_true.columns if col.startswith("start_")]
        if not cell_columns:
            raise InvalidSubmissionError("No starting board prediction columns found in ground truth data.")
        
        # Compute mean absolute error across all cells
        true_array = y_true[cell_columns].to_numpy(dtype=float)
        pred_array = y_pred[cell_columns].to_numpy(dtype=float)
        mae = np.mean(np.abs(true_array - pred_array))
        return mae

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        # Check that both submission and ground_truth are pandas DataFrames
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        
        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(
                f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows."
            )
        
        # Convert id column to string type in both submission and ground_truth; assume id is the first column
        submission[submission.columns[0]] = submission[submission.columns[0]].astype(str)
        ground_truth[ground_truth.columns[0]] = ground_truth[ground_truth.columns[0]].astype(str)
        
        # Sort both DataFrames by the id column
        submission = submission.sort_values(by=submission.columns[0]).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=ground_truth.columns[0]).reset_index(drop=True)
        
        # Check that the id columns are identical
        if not (submission[submission.columns[0]].values == ground_truth[ground_truth.columns[0]].values).all():
            raise InvalidSubmissionError("ID column values do not match between submission and ground truth. Please ensure the first column values are identical and in the same order.")
        
        # Define the required columns for a valid submission: id and start_0 to start_624 (625 cells for a 25x25 board)
        required_columns = {"id"}
        required_columns.update({f"start_{i}" for i in range(625)})
        submission_cols = set(submission.columns)
        
        missing_cols = required_columns - submission_cols
        extra_cols = submission_cols - required_columns
        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(sorted(missing_cols))}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(sorted(extra_cols))}.")

        return "Submission is valid."