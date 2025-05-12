from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class LibertyMutualGroupPropertyInspectionPredictionMetrics(CompetitionMetrics):
    """Metric class for Liberty Mutual Group Property Inspection Prediction competition using normalized Gini coefficient"""
    def __init__(self, value: str = "Hazard", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure the first column (Id) is treated as string and sort both dataframes by this column
        y_true[y_true.columns[0]] = y_true[y_true.columns[0]].astype(str)
        y_pred[y_pred.columns[0]] = y_pred[y_pred.columns[0]].astype(str)
        y_true = y_true.sort_values(by=y_true.columns[0]).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=y_pred.columns[0]).reset_index(drop=True)
        
        # Extract actual and predicted Hazard values as numpy arrays
        actual = y_true[self.value].to_numpy()
        pred = y_pred[self.value].to_numpy()

        def gini_coefficient(a: np.ndarray, p: np.ndarray) -> float:
            # Sort the actual values by descending order of predictions
            order = np.argsort(p)[::-1]
            a_sorted = a[order]
            n = len(a_sorted)
            cumulative_actual = np.cumsum(a_sorted)
            sum_actual = a_sorted.sum()
            if sum_actual == 0:
                return 0.0
            # The Lorenz curve sum adjusted by the ideal line
            gini_sum = cumulative_actual.sum() / sum_actual - (n + 1) / 2.0
            return gini_sum / n

        model_gini = gini_coefficient(actual, pred)
        perfect_gini = gini_coefficient(actual, actual)
        if perfect_gini == 0:
            return 0.0
        normalized_gini = model_gini / perfect_gini
        return normalized_gini

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        
        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Convert first column to string type and sort both dataframes by it (assumed to be the Id column)
        submission[submission.columns[0]] = submission[submission.columns[0]].astype(str)
        ground_truth[ground_truth.columns[0]] = ground_truth[ground_truth.columns[0]].astype(str)
        submission = submission.sort_values(by=submission.columns[0])
        ground_truth = ground_truth.sort_values(by=ground_truth.columns[0])

        # Check if the Id column values match
        if (submission[submission.columns[0]].values != ground_truth[ground_truth.columns[0]].values).any():
            raise InvalidSubmissionError("First column values do not match between submission and ground truth. Please ensure the first column values are identical.")

        sub_cols = set(submission.columns)
        true_cols = set(ground_truth.columns)

        missing_cols = true_cols - sub_cols
        extra_cols = sub_cols - true_cols

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."