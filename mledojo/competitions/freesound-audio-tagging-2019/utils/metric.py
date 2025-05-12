from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class FreesoundAudioTagging2019Metrics(CompetitionMetrics):
    """Metric class for Freesound Audio Tagging 2019 competition using label-weighted label-ranking average precision (lwlrap)"""
    def __init__(self, value: str = "lwlrap", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Convert the identifier column (assumed to be the first column, e.g., fname) to string
        y_true[y_true.columns[0]] = y_true[y_true.columns[0]].astype(str)
        y_pred[y_pred.columns[0]] = y_pred[y_pred.columns[0]].astype(str)
        
        # Sort both dataframes by the identifier column
        y_true = y_true.sort_values(by=y_true.columns[0]).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=y_pred.columns[0]).reset_index(drop=True)
        
        # Exclude the identifier column (fname) from calculations
        true_matrix = y_true.iloc[:, 1:].to_numpy()
        pred_matrix = y_pred.iloc[:, 1:].to_numpy()
        
        num_samples, num_labels = true_matrix.shape
        ap_per_label = np.zeros(num_labels)
        
        # Compute average precision for each label
        for j in range(num_labels):
            true_values = true_matrix[:, j]
            pred_scores = pred_matrix[:, j]
            
            # Sort indices in descending order of predicted scores
            sorted_indices = np.argsort(-pred_scores)
            true_sorted = true_values[sorted_indices]
            
            # Cumulative sum of true positives
            tp_cumsum = np.cumsum(true_sorted)
            # Compute precision at each rank position
            precision_at_hits = tp_cumsum / (np.arange(num_samples) + 1)
            
            # Only consider positions where the label is positive
            total_positives = np.sum(true_values)
            if total_positives > 0:
                ap = np.sum(precision_at_hits * true_sorted) / total_positives
            else:
                ap = 0.0
            ap_per_label[j] = ap
        
        # The overall lwlrap is the unweighted average of the per-label average precisions
        score = np.mean(ap_per_label)
        return score

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Convert the first column (identifier, e.g., fname) to string and sort both dataframes
        submission[submission.columns[0]] = submission[submission.columns[0]].astype(str)
        ground_truth[ground_truth.columns[0]] = ground_truth[ground_truth.columns[0]].astype(str)
        submission = submission.sort_values(by=submission.columns[0]).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=ground_truth.columns[0]).reset_index(drop=True)

        # Check if the identifier columns in both dataframes match exactly
        if not (submission[submission.columns[0]].values == ground_truth[ground_truth.columns[0]].values).all():
            raise InvalidSubmissionError("The first column values (e.g., file names) do not match between submission and ground truth. Please ensure they are identical.")

        submission_columns = set(submission.columns)
        ground_truth_columns = set(ground_truth.columns)
        missing_columns = ground_truth_columns - submission_columns
        extra_columns = submission_columns - ground_truth_columns

        if missing_columns:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_columns)}.")
        if extra_columns:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_columns)}.")

        return "Submission is valid."