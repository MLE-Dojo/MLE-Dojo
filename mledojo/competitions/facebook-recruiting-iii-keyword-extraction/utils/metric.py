from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class FacebookRecruitingIiiKeywordExtractionMetrics(CompetitionMetrics):
    """
    Metric class for Facebook Recruiting III Keyword Extraction competition using Mean F1-Score.
    """
    def __init__(self, value: str = "Tags", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure the first column (Id) is string type for both ground truth and submission
        id_col = y_true.columns[0]
        y_true[id_col] = y_true[id_col].astype(str)
        y_pred[id_col] = y_pred[id_col].astype(str)
        # Sort both DataFrames by the Id column
        y_true = y_true.sort_values(by=id_col).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=id_col).reset_index(drop=True)

        # Compute the mean F1-Score across all rows
        f1_scores = []
        # Iterate row by row
        for _, (true_row, pred_row) in enumerate(zip(y_true.iterrows(), y_pred.iterrows())):
            # Get the predicted and true tags as strings
            true_tags_str = true_row[1][self.value]
            pred_tags_str = pred_row[1][self.value]

            # Split the tags on whitespace into sets. Also strip any extra whitespace.
            true_tags = set(str(true_tags_str).split())
            pred_tags = set(str(pred_tags_str).split())

            # Special case: if both sets are empty, count as perfect score.
            if not true_tags and not pred_tags:
                f1_scores.append(1.0)
                continue

            # Calculate true positives as intersection size
            tp = len(true_tags.intersection(pred_tags))
            # Calculate precision and recall
            precision = tp / len(pred_tags) if pred_tags else 0.0
            recall = tp / len(true_tags) if true_tags else 0.0

            # Calculate F1 score; if both precision and recall are zero, F1 is zero.
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            f1_scores.append(f1)
            
        # Return the mean F1 score
        return np.mean(f1_scores)

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Convert first column (Id) to string type
        id_col = submission.columns[0]
        submission[id_col] = submission[id_col].astype(str)
        ground_truth[ground_truth.columns[0]] = ground_truth[ground_truth.columns[0]].astype(str)

        # Sort submission and ground truth by the Id column
        submission = submission.sort_values(by=id_col).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=ground_truth.columns[0]).reset_index(drop=True)

        # Check if the Id columns match exactly
        if (submission[id_col].values != ground_truth[ground_truth.columns[0]].values).any():
            raise InvalidSubmissionError("The 'Id' column values do not match between submission and ground truth. Please ensure the 'Id's are identical and in the same order.")

        required_cols = {submission.columns[0], self.value}
        sub_cols = set(submission.columns)
        true_cols = set(ground_truth.columns)

        missing_cols = required_cols - sub_cols
        extra_cols = sub_cols - required_cols

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."