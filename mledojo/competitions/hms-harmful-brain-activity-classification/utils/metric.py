from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class HmsHarmfulBrainActivityClassificationMetrics(CompetitionMetrics):
    """Metric class for HMS Harmful Brain Activity Classification competition using Kullback Liebler divergence."""
    def __init__(self, value: str = "seizure_vote", higher_is_better: bool = False):
        super().__init__(higher_is_better)
        self.value = value  # value is set to one of the vote columns (here, seizure_vote) for reference

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Convert the id column (eeg_id) to string type in both dataframes
        y_true[y_true.columns[0]] = y_true[y_true.columns[0]].astype(str)
        y_pred[y_pred.columns[0]] = y_pred[y_pred.columns[0]].astype(str)
        # Sort both dataframes by the id column
        y_true = y_true.sort_values(by=y_true.columns[0]).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=y_pred.columns[0]).reset_index(drop=True)
        
        vote_cols = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
        divergences = []
        epsilon = 1e-15

        # Calculate row-wise Kullback-Leibler divergence
        for idx in range(len(y_true)):
            # Get ground truth votes and convert to float
            true_vals = y_true.loc[idx, vote_cols].astype(float).values
            true_sum = true_vals.sum()
            # Normalize ground truth votes to probabilities; if sum==0, use zeros
            if true_sum == 0:
                true_probs = np.zeros_like(true_vals)
            else:
                true_probs = true_vals / true_sum

            # Get predicted probabilities and convert to float
            pred_probs = y_pred.loc[idx, vote_cols].astype(float).values
            # Clip predicted probabilities to avoid division by zero
            pred_probs = np.clip(pred_probs, epsilon, 1)
            # Compute KL divergence for the row; terms where true_prob==0 contribute 0
            kl_div = np.sum(np.where(true_probs > 0, true_probs * np.log(true_probs / pred_probs), 0))
            divergences.append(kl_div)

        # Return the average KL divergence over all rows
        return np.mean(divergences)

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        required_columns = {"eeg_id", "seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"}
        submission_cols = set(submission.columns)
        ground_truth_cols = set(ground_truth.columns)

        # Check that the number of rows is identical
        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(
                f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows."
            )

        # Ensure the id columns are strings and sort both dataframes by eeg_id
        submission["eeg_id"] = submission["eeg_id"].astype(str)
        ground_truth["eeg_id"] = ground_truth["eeg_id"].astype(str)
        submission = submission.sort_values(by="eeg_id")
        ground_truth = ground_truth.sort_values(by="eeg_id")

        # Check that the eeg_id values match exactly
        if not (submission["eeg_id"].values == ground_truth["eeg_id"].values).all():
            raise InvalidSubmissionError(
                "eeg_id values do not match between submission and ground truth. Please ensure the eeg_id values are identical and in the correct order."
            )

        missing_cols = required_columns - submission_cols
        extra_cols = submission_cols - required_columns

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        # Check that the probabilities in the vote columns sum to one for each row (within a small tolerance)
        vote_cols = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
        row_sums = submission[vote_cols].sum(axis=1)
        if not np.allclose(row_sums, 1, atol=1e-3):
            raise InvalidSubmissionError("The vote probabilities for each row must sum to one.")

        return "Submission is valid."