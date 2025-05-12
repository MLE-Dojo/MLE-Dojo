from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class BenchmarkBondTradePriceChallengeMetrics(CompetitionMetrics):
    """Metric class for Benchmark Bond Trade Price Challenge using weighted mean absolute error"""
    def __init__(self, value: str = "trade_price", higher_is_better: bool = False):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure that the dataframes have the 'id' column used for merging
        if 'id' not in y_true.columns or 'id' not in y_pred.columns:
            raise InvalidSubmissionError("Both ground truth and submission must contain 'id' column.")
            
        # Convert id column to string type to ensure proper alignment
        y_true['id'] = y_true['id'].astype(str)
        y_pred['id'] = y_pred['id'].astype(str)
        
        # Sort by 'id'
        y_true = y_true.sort_values(by='id').reset_index(drop=True)
        y_pred = y_pred.sort_values(by='id').reset_index(drop=True)
        
        # Check that the ids are identical
        if not (y_true['id'].values == y_pred['id'].values).all():
            raise InvalidSubmissionError("The 'id' column values do not match between ground truth and submission.")
        
        # Calculate weighted mean absolute error
        # Ground truth must contain 'weight' and the column to predict
        if 'weight' not in y_true.columns:
            raise InvalidSubmissionError("Ground truth is missing the required 'weight' column.")
        
        # Absolute errors multiplied by weights
        abs_errors = np.abs(y_true[self.value] - y_pred[self.value])
        weighted_error = np.sum(y_true['weight'] * abs_errors)
        total_weight = np.sum(y_true['weight'])
        score = weighted_error / total_weight
        return score

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        required_columns = {"id", self.value}
        gt_required_columns = {"id", self.value, "weight"}

        submission_cols = set(submission.columns)
        ground_truth_cols = set(ground_truth.columns)

        missing_submission = required_columns - submission_cols
        if missing_submission:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_submission)}.")
        missing_ground_truth = gt_required_columns - ground_truth_cols
        if missing_ground_truth:
            raise InvalidSubmissionError(f"Missing required columns in ground truth: {', '.join(missing_ground_truth)}.")

        # Convert id columns to string type
        submission['id'] = submission['id'].astype(str)
        ground_truth['id'] = ground_truth['id'].astype(str)

        # Sort by 'id'
        submission_sorted = submission.sort_values(by='id').reset_index(drop=True)
        ground_truth_sorted = ground_truth.sort_values(by='id').reset_index(drop=True)

        # Check if 'id' columns match
        if not (submission_sorted['id'].values == ground_truth_sorted['id'].values).all():
            raise InvalidSubmissionError("The 'id' column values do not match between submission and ground truth. Please ensure they are identical and in the same order.")

        return "Submission is valid."