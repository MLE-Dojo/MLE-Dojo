from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError
from scipy.stats import spearmanr

class DraperSatelliteImageChronologyMetrics(CompetitionMetrics):
    """Metric class for Draper Satellite Image Chronology competition using Mean Spearman Correlation"""
    def __init__(self, value: str = "day", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure the id column 'setId' is of type string
        y_true['setId'] = y_true['setId'].astype(str)
        y_pred['setId'] = y_pred['setId'].astype(str)
        
        # Sort by 'setId'
        y_true = y_true.sort_values(by='setId').reset_index(drop=True)
        y_pred = y_pred.sort_values(by='setId').reset_index(drop=True)
        
        correlations = []
        # Iterate over each row (each set of 5 images)
        for i in range(len(y_true)):
            true_days_str = y_true.iloc[i][self.value]
            pred_days_str = y_pred.iloc[i][self.value]
            
            try:
                true_days = [int(x) for x in true_days_str.strip().split()]
                pred_days = [int(x) for x in pred_days_str.strip().split()]
            except Exception as e:
                raise InvalidSubmissionError(f"Error parsing the '{self.value}' column on row {i}: {e}")
            
            if len(true_days) != len(pred_days):
                raise InvalidSubmissionError(f"Row {i}: The number of predicted days does not match ground truth.")

            # Calculate Spearman correlation for the set
            corr, _ = spearmanr(true_days, pred_days)
            correlations.append(corr)
        
        # Calculate and return the mean Spearman correlation coefficient
        return np.mean(correlations)

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        required_columns = {"setId", self.value}
        submission_columns = set(submission.columns)
        ground_truth_columns = set(ground_truth.columns)

        if not required_columns.issubset(submission_columns):
            missing = required_columns - submission_columns
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing)}.")

        if not required_columns.issubset(ground_truth_columns):
            missing = required_columns - ground_truth_columns
            raise InvalidSubmissionError(f"Missing required columns in ground truth: {', '.join(missing)}.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}).")

        # Convert 'setId' to string in both submission and ground_truth
        submission['setId'] = submission['setId'].astype(str)
        ground_truth['setId'] = ground_truth['setId'].astype(str)

        # Sort the submission and ground_truth by 'setId'
        submission_sorted = submission.sort_values(by='setId').reset_index(drop=True)
        ground_truth_sorted = ground_truth.sort_values(by='setId').reset_index(drop=True)

        # Check if 'setId' values are identical across submission and ground truth
        if not (submission_sorted['setId'].values == ground_truth_sorted['setId'].values).all():
            raise InvalidSubmissionError("The 'setId' values in submission do not match those in ground truth in order. Please ensure they are identical.")

        return "Submission is valid."