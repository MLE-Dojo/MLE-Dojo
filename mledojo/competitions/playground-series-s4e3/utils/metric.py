from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError
from sklearn.metrics import roc_auc_score

class PlaygroundSeriesS4e3Metrics(CompetitionMetrics):
    """Metric class for playground-series-s4e3 competition using average AUC across defect categories"""
    def __init__(self, value: str = "AUC", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value  # This attribute is symbolic since score is computed over multiple columns

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure the id column is treated as string and sort by id for both dataframes
        y_true["id"] = y_true["id"].astype(str)
        y_pred["id"] = y_pred["id"].astype(str)
        y_true = y_true.sort_values(by="id").reset_index(drop=True)
        y_pred = y_pred.sort_values(by="id").reset_index(drop=True)

        # Define the defect columns
        defect_cols = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]
        auc_scores = []
        for col in defect_cols:
            # Calculate ROC AUC for each defect category
            auc = roc_auc_score(y_true[col], y_pred[col])
            auc_scores.append(auc)

        # Return the average AUC across all defect categories
        return np.mean(auc_scores)

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Ensure the id column is of string type and sort by id
        submission["id"] = submission["id"].astype(str)
        ground_truth["id"] = ground_truth["id"].astype(str)
        submission = submission.sort_values(by="id").reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by="id").reset_index(drop=True)

        # Check if id columns match
        if not (submission["id"].values == ground_truth["id"].values).all():
            raise InvalidSubmissionError("ID column values do not match between submission and ground truth. Please ensure the ID values are identical and in the same order.")

        expected_columns = {"id", "Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"}
        submission_cols = set(submission.columns)
        ground_truth_cols = set(ground_truth.columns)

        missing_cols = ground_truth_cols - submission_cols
        extra_cols = submission_cols - ground_truth_cols

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."