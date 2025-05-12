from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError
from sklearn.metrics import f1_score

class Imet2020Fgvc7Metrics(CompetitionMetrics):
    """Metric class for imet-2020-fgvc7 competition using micro averaged F1 score for multi-label classification."""
    def __init__(self, value: str = "attribute_ids", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure the first column (id) is of string type and sort by it
        id_col_true = y_true.columns[0]
        id_col_pred = y_pred.columns[0]
        y_true[id_col_true] = y_true[id_col_true].astype(str)
        y_pred[id_col_pred] = y_pred[id_col_pred].astype(str)
        y_true = y_true.sort_values(by=id_col_true).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=id_col_pred).reset_index(drop=True)

        # Extract the true and predicted attribute_ids as strings.
        true_labels = y_true[self.value].astype(str)
        pred_labels = y_pred[self.value].astype(str)

        # Build the set of all unique attribute labels from ground truth.
        all_classes = set()
        for labels in true_labels:
            tokens = labels.split()
            all_classes.update(tokens)
        all_classes = sorted(list(all_classes))

        # Helper function: Convert a series of label strings into a binary matrix.
        # This handles cases where the number of attribute_ids varies between samples
        def get_binary_matrix(series: pd.Series, classes: list) -> np.ndarray:
            binary_matrix = []
            for row in series:
                tokens = row.split()
                binary_row = [1 if cls in tokens else 0 for cls in classes]
                binary_matrix.append(binary_row)
            return np.array(binary_matrix)

        y_true_binary = get_binary_matrix(true_labels, all_classes)
        y_pred_binary = get_binary_matrix(pred_labels, all_classes)

        # Compute micro averaged F1 score.
        # This metric handles imbalanced datasets where the number of attributes varies between samples
        score = f1_score(y_true_binary, y_pred_binary, average='micro')
        return score

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Convert first column to string type
        sub_id = submission.columns[0]
        true_id = ground_truth.columns[0]
        submission[sub_id] = submission[sub_id].astype(str)
        ground_truth[true_id] = ground_truth[true_id].astype(str)

        # Sort the submission and ground truth by the first column
        submission = submission.sort_values(by=sub_id).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=true_id).reset_index(drop=True)

        # Check if first columns are identical
        if not np.array_equal(submission[sub_id].values, ground_truth[true_id].values):
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