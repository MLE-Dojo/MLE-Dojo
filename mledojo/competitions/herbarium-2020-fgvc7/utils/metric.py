from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError
from sklearn.metrics import f1_score

class Herbarium2020Fgvc7Metrics(CompetitionMetrics):
    """Metric class for Herbarium 2020 FGVC7 competition using Macro F1 Score."""
    def __init__(self, value: str = "Predicted", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Convert the first column (Id) to string type and sort both DataFrames by Id
        y_true[y_true.columns[0]] = y_true[y_true.columns[0]].astype(str)
        y_pred[y_pred.columns[0]] = y_pred[y_pred.columns[0]].astype(str)
        y_true = y_true.sort_values(by=y_true.columns[0]).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=y_pred.columns[0]).reset_index(drop=True)
        
        # Expected ground truth label column is 'category_id'
        true_labels = y_true[self.value]
        pred_labels = y_pred[self.value]
        # Calculate macro F1 score
        score = f1_score(true_labels, pred_labels, average="macro")
        return score

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Convert first column (Id) to string type
        submission[submission.columns[0]] = submission[submission.columns[0]].astype(str)
        ground_truth[ground_truth.columns[0]] = ground_truth[ground_truth.columns[0]].astype(str)
        # Sort the submission and ground truth by the first column (Id)
        submission = submission.sort_values(by=submission.columns[0]).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=ground_truth.columns[0]).reset_index(drop=True)

        # Check if first columns (Id) are identical
        if not (submission[submission.columns[0]].values == ground_truth[ground_truth.columns[0]].values).all():
            raise InvalidSubmissionError("First column values (Id) do not match between submission and ground truth. Please ensure the first column values are identical.")

        # Check necessary columns: submission should have "Id" and self.value, ground truth should have "Id" and self.value
        submission_required = {submission.columns[0], self.value}
        ground_truth_required = {ground_truth.columns[0], self.value}

        sub_cols = set(submission.columns)
        true_cols = set(ground_truth.columns)

        missing_sub_cols = submission_required - sub_cols
        missing_true_cols = ground_truth_required - true_cols

        if missing_sub_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_sub_cols)}.")
        if missing_true_cols:
            raise InvalidSubmissionError(f"Missing required columns in ground truth: {', '.join(missing_true_cols)}.")

        extra_sub_cols = sub_cols - submission_required
        # Optionally, you can warn about extra columns but not necessarily raise an error.
        # For strict checking uncomment the lines below:
        # if extra_sub_cols:
        #     raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_sub_cols)}.")

        return "Submission is valid."