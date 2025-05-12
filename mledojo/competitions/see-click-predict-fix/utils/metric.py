from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class SeeClickPredictFixMetrics(CompetitionMetrics):
    """
    Metric class for See Click Predict Fix competition using Root Mean Squared Logarithmic Error (RMSLE).
    The submission file should have the following columns:
    id,num_views,num_votes,num_comments
    Lower RMSLE indicates a better model.
    """
    def __init__(self, value: str = "num_views", higher_is_better: bool = False):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure the id column (first column) is of string type for proper sorting
        id_col = y_true.columns[0]
        y_true[id_col] = y_true[id_col].astype(str)
        y_pred[id_col] = y_pred[id_col].astype(str)
        y_true = y_true.sort_values(by=id_col).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=id_col).reset_index(drop=True)
        
        # Define target columns
        target_cols = ["num_views", "num_votes", "num_comments"]
        if not all(col in y_true.columns for col in target_cols):
            raise InvalidSubmissionError("Ground truth is missing one or more required target columns: num_views, num_votes, num_comments.")
        if not all(col in y_pred.columns for col in target_cols):
            raise InvalidSubmissionError("Submission is missing one or more required target columns: num_views, num_votes, num_comments.")
        
        # Compute squared logarithmic errors for all target columns
        squared_log_errors = []
        for col in target_cols:
            # Using np.log1p ensures that negative values will cause an error if not handled;
            # assuming predictions and true values are non-negative.
            diff = np.log1p(y_pred[col].to_numpy()) - np.log1p(y_true[col].to_numpy())
            squared_log_errors.append(diff ** 2)
        # Flatten all errors into a single array and compute the mean
        all_errors = np.concatenate(squared_log_errors)
        mean_squared_log_error = np.mean(all_errors)
        rmsle = np.sqrt(mean_squared_log_error)
        return rmsle

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        expected_columns = set(["id", "num_views", "num_votes", "num_comments"])
        sub_columns = set(submission.columns)
        gt_columns = set(ground_truth.columns)

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Convert id column (first column) to string type and sort both submission and ground truth by id
        id_col = submission.columns[0]
        submission[id_col] = submission[id_col].astype(str)
        ground_truth[id_col] = ground_truth[id_col].astype(str)
        submission = submission.sort_values(by=id_col).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=id_col).reset_index(drop=True)

        if not np.array_equal(submission[id_col].values, ground_truth[id_col].values):
            raise InvalidSubmissionError("The id column values do not match between submission and ground truth. Please ensure the id values are identical and in the same order.")

        missing_cols = expected_columns - sub_columns
        extra_cols = sub_columns - expected_columns
        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."