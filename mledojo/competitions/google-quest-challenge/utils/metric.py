from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError
from scipy.stats import spearmanr

class GoogleQuestChallengeMetrics(CompetitionMetrics):
    """
    Metric class for the google-quest-challenge competition.
    The score is computed as the mean of the Spearman's rank correlation coefficients
    for each target column.
    """
    def __init__(self, value: str = "mean_spearman", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Convert the id column (first column, 'qa_id') to string type
        id_col_true = y_true.columns[0]
        id_col_pred = y_pred.columns[0]
        y_true[id_col_true] = y_true[id_col_true].astype(str)
        y_pred[id_col_pred] = y_pred[id_col_pred].astype(str)
        
        # Sort both DataFrames by the id column
        y_true = y_true.sort_values(by=id_col_true).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=id_col_pred).reset_index(drop=True)
        
        # Compute Spearman's correlation for each target column (all columns except the first one)
        correlations = []
        for col in y_true.columns[1:]:
            true_values = y_true[col]
            pred_values = y_pred[col]
            corr = spearmanr(true_values, pred_values).correlation
            # Handle cases where spearmanr returns nan (e.g., constant values)
            if np.isnan(corr):
                corr = 0.0
            correlations.append(corr)
        
        # Return the mean Spearman's correlation coefficient
        if correlations:
            return np.mean(correlations)
        return 0.0

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Convert id column (first column) to string type
        id_col_sub = submission.columns[0]
        id_col_true = ground_truth.columns[0]
        submission[id_col_sub] = submission[id_col_sub].astype(str)
        ground_truth[id_col_true] = ground_truth[id_col_true].astype(str)
        
        # Sort the submission and ground truth by the id column
        submission = submission.sort_values(by=id_col_sub).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=id_col_true).reset_index(drop=True)

        # Check if the id columns are identical
        if not np.array_equal(submission[id_col_sub].values, ground_truth[id_col_true].values):
            raise InvalidSubmissionError("ID column values do not match between submission and ground truth. Please ensure the first column values are identical.")

        sub_cols = set(submission.columns)
        true_cols = set(ground_truth.columns)

        missing_cols = true_cols - sub_cols
        extra_cols = sub_cols - true_cols

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."