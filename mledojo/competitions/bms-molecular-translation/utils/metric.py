from typing import Any
import pandas as pd
import numpy as np
import Levenshtein
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class BmsMolecularTranslationMetrics(CompetitionMetrics):
    """Metric class for BMS Molecular Translation competition using mean Levenshtein distance."""
    def __init__(self, value: str = "InChI", higher_is_better: bool = False):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure the id column is a string for both true and prediction DataFrames
        id_column_true = y_true.columns[0]  # image_id column
        id_column_pred = y_pred.columns[0]  # image_id column
        y_true[id_column_true] = y_true[id_column_true].astype(str)
        y_pred[id_column_pred] = y_pred[id_column_pred].astype(str)
        
        # Sort the dataframes by the id column ("image_id") so that rows align.
        y_true = y_true.sort_values(by=id_column_true).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=id_column_pred).reset_index(drop=True)
        
        # Compute the Levenshtein distance for each prediction vs. truth
        # This measures the string edit distance between predicted InChI strings and ground truth
        distances = []
        for true_str, pred_str in zip(y_true[self.value], y_pred[self.value]):
            distances.append(Levenshtein.distance(true_str, pred_str))
        
        # Return the mean Levenshtein distance as the score
        # Lower values are better (less distance between prediction and truth)
        return np.mean(distances)

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Ensure the id columns are treated as strings
        submission[submission.columns[0]] = submission[submission.columns[0]].astype(str)
        ground_truth[ground_truth.columns[0]] = ground_truth[ground_truth.columns[0]].astype(str)
        
        # Sort by the id column ("image_id")
        submission = submission.sort_values(by=submission.columns[0]).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=ground_truth.columns[0]).reset_index(drop=True)

        # Check if the id columns match exactly between submission and ground truth
        if (submission[submission.columns[0]].values != ground_truth[ground_truth.columns[0]].values).any():
            raise InvalidSubmissionError("The id column values do not match between submission and ground truth. Please ensure the first column values are identical.")

        submission_cols = set(submission.columns)
        ground_truth_cols = set(ground_truth.columns)

        missing_cols = ground_truth_cols - submission_cols
        extra_cols = submission_cols - ground_truth_cols

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."