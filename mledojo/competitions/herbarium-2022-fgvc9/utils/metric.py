from typing import Any
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class Herbarium2022Fgvc9Metrics(CompetitionMetrics):
    """Metric class for Herbarium 2022 FGVC9 competition using macro F1 score."""
    def __init__(self, value: str = "Predicted", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Convert the 'Id' column to string type for both DataFrames
        y_true[y_true.columns[0]] = y_true[y_true.columns[0]].astype(str)
        y_pred[y_pred.columns[0]] = y_pred[y_pred.columns[0]].astype(str)
        # Sort the DataFrames by the 'Id' column (first column)
        y_true = y_true.sort_values(by=y_true.columns[0]).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=y_pred.columns[0]).reset_index(drop=True)
        
        # Calculate macro F1 score
        # In "macro" F1 a separate F1 score is calculated for each species value and then averaged
        macro_f1 = f1_score(
            y_true[self.value], 
            y_pred[self.value], 
            average='macro'
        )
        
        return macro_f1

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Ensure the first column (Id) is of string type and sort both submission and ground truth by it
        submission[submission.columns[0]] = submission[submission.columns[0]].astype(str)
        ground_truth[ground_truth.columns[0]] = ground_truth[ground_truth.columns[0]].astype(str)
        submission = submission.sort_values(by=submission.columns[0]).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=ground_truth.columns[0]).reset_index(drop=True)

        # Check if the first column (Id) matches in both submission and ground truth
        if not (submission[submission.columns[0]].values == ground_truth[ground_truth.columns[0]].values).all():
            raise InvalidSubmissionError("First column values (Id) do not match between submission and ground truth. Please ensure the Id values are identical and in the same order.")

        # Validate that the submission contains exactly the required columns: 'Id' and the metric column.
        required_columns = {submission.columns[0], self.value}
        submission_columns = set(submission.columns)
        ground_truth_columns = set(ground_truth.columns)

        missing_cols = required_columns - submission_columns
        extra_cols = submission_columns - ground_truth_columns

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."