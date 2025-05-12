from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError


class Nomad2018PredictTransparentConductorsMetrics(CompetitionMetrics):
    """Metric class for Nomad2018PredictTransparentConductors competition using RMSLE"""

    def __init__(self, value: str = "formation_energy_ev_natom", higher_is_better: bool = False):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Sort both dataframes by first column before calculating score
        y_true = y_true.sort_values(by=y_true.columns[0]).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=y_pred.columns[0]).reset_index(drop=True)
        columns = ["formation_energy_ev_natom", "bandgap_energy_ev"]
        rmsle_scores = []

        for col in columns:
            log_true = np.log(y_true[col] + 1)
            log_pred = np.log(y_pred[col] + 1)
            rmsle = np.sqrt(np.mean((log_true - log_pred) ** 2))
            rmsle_scores.append(rmsle)

        return np.mean(rmsle_scores)

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError(
                "Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame."
            )
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError(
                "Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame."
            )

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(
                f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows."
            )

        # Sort the submission and ground truth by the first column
        submission = submission.sort_values(by=submission.columns[0])
        ground_truth = ground_truth.sort_values(by=ground_truth.columns[0])

        # Check if first columns are identical
        if (submission[submission.columns[0]].values != (ground_truth[ground_truth.columns[0]].values)).any():
            raise InvalidSubmissionError(
                "First column values do not match between submission and ground truth. Please ensure the first column values are identical."
            )

        required_columns = {"id", "formation_energy_ev_natom", "bandgap_energy_ev"}
        submission_cols = set(submission.columns)
        missing_cols = required_columns - submission_cols
        extra_cols = submission_cols - required_columns

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."
