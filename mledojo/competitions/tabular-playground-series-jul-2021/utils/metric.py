from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class TabularPlaygroundSeriesJul2021Metrics(CompetitionMetrics):
    """
    Metric class for the Tabular Playground Series Jul 2021 competition.
    Evaluation is based on the mean column-wise root mean squared logarithmic error (RMSLE)
    over the three target columns: 
      - target_carbon_monoxide
      - target_benzene
      - target_nitrogen_oxides
    Lower score is better.
    """
    def __init__(self, value: str = "target_carbon_monoxide", higher_is_better: bool = False):
        super().__init__(higher_is_better)
        # value is set to the first target column by convention
        self.value = value
        # Expected columns in submissions and ground truth
        self.expected_columns = {
            "date_time",
            "target_carbon_monoxide",
            "target_benzene",
            "target_nitrogen_oxides"
        }
    
    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure the first column (id column "date_time") is of type string and sort by it
        id_col = y_true.columns[0]
        y_true[id_col] = y_true[id_col].astype(str)
        y_pred[id_col] = y_pred[id_col].astype(str)
        y_true = y_true.sort_values(by=id_col).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=id_col).reset_index(drop=True)
        
        target_cols = ["target_carbon_monoxide", "target_benzene", "target_nitrogen_oxides"]
        rmsle_scores = []
        
        for col in target_cols:
            # Calculate RMSLE for each target column
            # Ensure no negative values before applying log1p
            if (y_pred[col] < 0).any() or (y_true[col] < 0).any():
                raise ValueError(f"Negative values found in column {col}. Predictions and actuals must be non-negative for RMSLE.")
            log_pred = np.log1p(y_pred[col].values)
            log_true = np.log1p(y_true[col].values)
            rmsle = np.sqrt(np.mean((log_pred - log_true) ** 2))
            rmsle_scores.append(rmsle)
        
        # Return the mean of the RMSLE scores over the target columns
        return float(np.mean(rmsle_scores))
    
    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        
        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")
        
        # The first column is used as the id column; in this competition it is "date_time"
        id_col = submission.columns[0]
        submission[id_col] = submission[id_col].astype(str)
        ground_truth[id_col] = ground_truth[id_col].astype(str)
        
        # Sort both submission and ground truth based on the id column
        submission = submission.sort_values(by=id_col).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=id_col).reset_index(drop=True)
        
        # Check if the id columns match exactly
        if not np.array_equal(submission[id_col].values, ground_truth[id_col].values):
            raise InvalidSubmissionError("The values in the id column do not match between submission and ground truth. Ensure that the 'date_time' column is identical in both files.")

        # Check if submission has all required columns and no extra columns
        sub_columns = set(submission.columns)
        missing_cols = self.expected_columns - sub_columns
        extra_cols = sub_columns - self.expected_columns
        
        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")
        
        return "Submission is valid."