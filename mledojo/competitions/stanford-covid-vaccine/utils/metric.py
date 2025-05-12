from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class StanfordCovidVaccineMetrics(CompetitionMetrics):
    """Metric class for the stanford-covid-vaccine competition using Mean Columnwise RMSE (MCRMSE)"""
    def __init__(self, value: str = "reactivity", higher_is_better: bool = False):
        super().__init__(higher_is_better)
        # 'value' is set to reactivity, one of the scored targets
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure the id column (first column, "id_seqpos") is of string type
        id_col = y_true.columns[0]
        y_true[id_col] = y_true[id_col].astype(str)
        y_pred[id_col] = y_pred[id_col].astype(str)
        # Sort by the id column
        y_true = y_true.sort_values(by=id_col).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=id_col).reset_index(drop=True)

        # Define the target columns to be scored
        target_columns = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

        rmses = []
        for col in target_columns:
            # Compute root mean squared error for each column
            diff = y_true[col] - y_pred[col]
            rmse = np.sqrt(np.mean(diff ** 2))
            rmses.append(rmse)

        mcrmse = np.mean(rmses)
        return mcrmse

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Convert the id column to string type
        id_col_sub = submission.columns[0]
        id_col_true = ground_truth.columns[0]
        submission[id_col_sub] = submission[id_col_sub].astype(str)
        ground_truth[id_col_true] = ground_truth[id_col_true].astype(str)
        # Sort both submission and ground truth by their id column
        submission = submission.sort_values(by=id_col_sub).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=id_col_true).reset_index(drop=True)

        # Check if the id columns are identical
        if not (submission[id_col_sub].values == ground_truth[id_col_true].values).all():
            raise InvalidSubmissionError("The identifier column values do not match between submission and ground truth. Please ensure the first column values are identical and in the same order.")

        required_cols = {"id_seqpos", "reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"}
        missing_cols = required_cols - set(submission.columns)
        extra_cols = set(submission.columns) - required_cols

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."