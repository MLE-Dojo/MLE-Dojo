from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class WhaleCategorizationPlaygroundMetrics(CompetitionMetrics):
    """
    Metric class for Whale Categorization Playground competition using Mean Average Precision at 5 (MAP@5).
    """
    def __init__(self, value: str = "Id", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure the first column (Image) is string and sort by it
        y_true[y_true.columns[0]] = y_true[y_true.columns[0]].astype(str)
        y_pred[y_pred.columns[0]] = y_pred[y_pred.columns[0]].astype(str)
        y_true = y_true.sort_values(by=y_true.columns[0]).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=y_pred.columns[0]).reset_index(drop=True)
        
        # Merge on the image id (first column) to align the rows
        merged = pd.merge(y_true, y_pred, on=y_true.columns[0], suffixes=('_true', '_pred'))
        
        def apk(actual, predicted, k=5):
            """
            Computes the average precision at k.
            actual: the true label (string) - a single whale ID
            predicted: list of predicted labels (up to 5 whale IDs)
            """
            if not predicted:
                return 0.0
            predicted = predicted[:k]
            try:
                index = predicted.index(actual)
                return 1.0 / (index + 1)
            except ValueError:
                return 0.0

        # Calculate MAP@5
        scores = []
        true_label_col = self.value + "_true"
        pred_label_col = self.value + "_pred"

        for _, row in merged.iterrows():
            true_label = row[true_label_col]  # This is a single whale ID
            pred_str = row[pred_label_col]
            # Split the predictions by whitespace (up to 5 whale IDs)
            pred_list = pred_str.split()
            score = apk(true_label, pred_list, k=5)
            scores.append(score)
        
        map5 = np.mean(scores)
        return map5

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Convert the 'Image' column to string type and sort both dataframes by it
        image_col = ground_truth.columns[0]
        submission[image_col] = submission[image_col].astype(str)
        ground_truth[image_col] = ground_truth[image_col].astype(str)
        submission = submission.sort_values(by=image_col).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=image_col).reset_index(drop=True)

        # Check if 'Image' columns match exactly
        if not (submission[image_col].values == ground_truth[image_col].values).all():
            raise InvalidSubmissionError("The 'Image' column values do not match between submission and ground truth. Please ensure they are identical.")

        # Check that required columns exist
        required_cols = set(ground_truth.columns)
        submission_cols = set(submission.columns)
        missing_cols = required_cols - submission_cols
        extra_cols = submission_cols - required_cols

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."