from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class TheWintonStockMarketChallengeMetrics(CompetitionMetrics):
    """Metric class for Winton Stock Market Challenge using Weighted Mean Absolute Error (WMAE)"""
    def __init__(self, value: str = "Predicted", higher_is_better: bool = False):
        # For error metrics like MAE, lower is better so higher_is_better is False.
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure the 'Id' column is of string type
        y_true['Id'] = y_true['Id'].astype(str)
        y_pred['Id'] = y_pred['Id'].astype(str)

        # Sort both dataframes by the 'Id' column
        y_true = y_true.sort_values(by='Id').reset_index(drop=True)
        y_pred = y_pred.sort_values(by='Id').reset_index(drop=True)

        # Check that both dataframes have the required prediction column
        if self.value not in y_pred.columns:
            raise InvalidSubmissionError(f"Submission DataFrame must contain the column '{self.value}'.")

        # Compute absolute errors
        abs_errors = np.abs(y_true[self.value] - y_pred[self.value])
        
        # Initialize an array for weights with same length as the prediction set
        weights = np.zeros(len(y_true))
        for i, id_val in enumerate(y_true['Id']):
            try:
                # Split the id, expecting format like 'windowIndex_returnIndex'
                parts = id_val.split('_')
                if len(parts) != 2:
                    raise ValueError
                return_idx = int(parts[1])
            except ValueError:
                raise InvalidSubmissionError("Id column values must be in the format '<window>_<return_index>' (e.g., '1_1').")
            
            # Assign weights based on return index
            if 1 <= return_idx <= 60:  # Intraday returns (Ret_121 through Ret_180)
                weights[i] = y_true.loc[i, "Weight_Intraday"]
            elif return_idx in [61, 62]:  # Daily returns (Ret_PlusOne and Ret_PlusTwo)
                weights[i] = y_true.loc[i, "Weight_Daily"]
            else:
                raise InvalidSubmissionError("Return index in Id must be between 1 and 62.")

        # Calculate the Weighted Mean Absolute Error (WMAE)
        wmae = np.sum(weights * abs_errors) / len(y_true)
        return wmae

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        # Check that both submission and ground truth are pandas DataFrames
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame.")

        # Check that submission contains required columns: 'Id' and 'Predicted'
        required_submission_cols = {"Id", "Predicted"}
        if not required_submission_cols.issubset(set(submission.columns)):
            missing = required_submission_cols - set(submission.columns)
            raise InvalidSubmissionError(f"Submission is missing the following required column(s): {', '.join(missing)}.")

        # Check that ground truth contains required columns
        required_ground_truth_cols = {"Id", "Predicted", "Weight_Intraday", "Weight_Daily"}
        if not required_ground_truth_cols.issubset(set(ground_truth.columns)):
            missing = required_ground_truth_cols - set(ground_truth.columns)
            raise InvalidSubmissionError(f"Ground truth is missing the following required column(s): {', '.join(missing)}.")

        # Check for missing values in submission
        if submission.isnull().any().any():
            raise InvalidSubmissionError("Submission contains missing values. All values must be provided.")

        # Convert the 'Id' column of both DataFrames to string type
        submission["Id"] = submission["Id"].astype(str)
        ground_truth["Id"] = ground_truth["Id"].astype(str)

        # Sort both DataFrames by the 'Id' column
        submission_sorted = submission.sort_values(by="Id").reset_index(drop=True)
        ground_truth_sorted = ground_truth.sort_values(by="Id").reset_index(drop=True)

        # Check if the 'Id' columns of submission and ground truth match exactly
        if not (submission_sorted["Id"].values == ground_truth_sorted["Id"].values).all():
            raise InvalidSubmissionError("The 'Id' values in submission do not match those in ground truth.")

        # Check that all Id values follow the expected format
        for id_val in submission["Id"]:
            try:
                parts = id_val.split('_')
                if len(parts) != 2:
                    raise ValueError
                window_idx = int(parts[0])
                return_idx = int(parts[1])
                if return_idx < 1 or return_idx > 62:
                    raise ValueError
            except ValueError:
                raise InvalidSubmissionError(f"Invalid Id format: {id_val}. Expected format is '<window>_<return_index>' where return_index is between 1 and 62.")

        return "Submission is valid."