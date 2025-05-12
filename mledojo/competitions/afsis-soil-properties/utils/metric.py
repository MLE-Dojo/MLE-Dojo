from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class AfsisSoilPropertiesMetrics(CompetitionMetrics):
    """
    Metric class for afsis-soil-properties competition which uses the Mean Columnwise 
    Root Mean Squared Error (MCRMSE) as the evaluation metric.
    """
    def __init__(self, value: str = "MCRMSE", higher_is_better: bool = False):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure the identifier column is of string type
        id_col = y_true.columns[0]
        y_true[id_col] = y_true[id_col].astype(str)
        y_pred[id_col] = y_pred[id_col].astype(str)
        
        # Sort both dataframes by the identifier column
        y_true = y_true.sort_values(by=id_col).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=id_col).reset_index(drop=True)
        
        # List of target columns for soil property predictions
        target_columns = ["Ca", "P", "pH", "SOC", "Sand"]
        
        # Compute RMSE for each target column
        rmse_list = []
        for col in target_columns:
            if col not in y_true.columns or col not in y_pred.columns:
                raise InvalidSubmissionError(f"Missing target column '{col}' in submission or ground truth.")
            mse = np.mean((y_true[col] - y_pred[col]) ** 2)
            rmse = np.sqrt(mse)
            rmse_list.append(rmse)
        
        # Compute the mean of RMSEs
        mcrmse = np.mean(rmse_list)
        return mcrmse

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        required_cols = {"PIDN", "Ca", "P", "pH", "SOC", "Sand"}
        
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")
        
        # Ensure the identifier (first column) is treated as string and sort both frames by PIDN
        id_col = "PIDN"
        if id_col not in submission.columns or id_col not in ground_truth.columns:
            raise InvalidSubmissionError(f"Identifier column '{id_col}' is missing from submission or ground truth.")

        submission[id_col] = submission[id_col].astype(str)
        ground_truth[id_col] = ground_truth[id_col].astype(str)
        
        submission = submission.sort_values(by=id_col).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=id_col).reset_index(drop=True)
        
        # Check if identifier columns match exactly
        if not np.array_equal(submission[id_col].values, ground_truth[id_col].values):
            raise InvalidSubmissionError("Identifier column values do not match between submission and ground truth. Please ensure the 'PIDN' column values are identical.")

        sub_cols = set(submission.columns)
        true_cols = required_cols
        missing_cols = true_cols - sub_cols
        extra_cols = sub_cols - true_cols

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."