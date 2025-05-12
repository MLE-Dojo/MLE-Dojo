from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

class PlantPathology2021Fgvc8Metrics(CompetitionMetrics):
    """Metric class for the Plant Pathology 2021-FGVC8 competition using Mean F1-Score."""
    def __init__(self, value: str = "labels", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure the identifier column (first column, "image") is of string type
        y_true[y_true.columns[0]] = y_true[y_true.columns[0]].astype(str)
        y_pred[y_pred.columns[0]] = y_pred[y_pred.columns[0]].astype(str)
        
        # Sort both dataframes by the identifier column
        y_true = y_true.sort_values(by=y_true.columns[0]).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=y_pred.columns[0]).reset_index(drop=True)
        
        # Convert the string of labels into lists by splitting on whitespace
        true_labels = y_true[self.value].apply(lambda x: x.split())
        pred_labels = y_pred[self.value].apply(lambda x: x.split())
        
        # Binarize the multi-label outputs
        mlb = MultiLabelBinarizer()
        all_labels = list(true_labels) + list(pred_labels)
        mlb.fit(all_labels)
        y_true_bin = mlb.transform(true_labels)
        y_pred_bin = mlb.transform(pred_labels)
        
        # Compute the mean F1 score across samples
        score = f1_score(y_true_bin, y_pred_bin, average='samples')
        return score

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(
                f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows."
            )

        # Convert identifier columns to string type
        submission[submission.columns[0]] = submission[submission.columns[0]].astype(str)
        ground_truth[ground_truth.columns[0]] = ground_truth[ground_truth.columns[0]].astype(str)
        
        # Sort the submission and ground truth by the identifier column
        submission = submission.sort_values(by=submission.columns[0]).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=ground_truth.columns[0]).reset_index(drop=True)

        # Check if identifier columns are identical
        if (submission[submission.columns[0]].values != ground_truth[ground_truth.columns[0]].values).any():
            raise InvalidSubmissionError("The identifier column values do not match between submission and ground truth. Please ensure they are identical.")

        submission_cols = set(submission.columns)
        truth_cols = set(ground_truth.columns)
        missing_cols = truth_cols - submission_cols
        extra_cols = submission_cols - truth_cols

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."