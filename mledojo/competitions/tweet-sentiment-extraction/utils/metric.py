from typing import Any
import pandas as pd
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class TweetSentimentExtractionMetrics(CompetitionMetrics):
    """Metric class for Tweet Sentiment Extraction competition using the word-level Jaccard index."""
    def __init__(self, value: str = "selected_text", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def jaccard(self, str1: str, str2: str) -> float:
        a = set(str1.lower().split())
        b = set(str2.lower().split())
        intersection = a.intersection(b)
        union = a.union(b)
        if len(union) == 0:
            return 0.0
        return float(len(intersection)) / len(union)

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure the id column ('textID') is of type string
        id_col = "textID"
        y_true[id_col] = y_true[id_col].astype(str)
        y_pred[id_col] = y_pred[id_col].astype(str)
        
        # Sort dataframes by the id column
        y_true = y_true.sort_values(by=id_col).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=id_col).reset_index(drop=True)
        
        # Calculate the jaccard score for each entry and compute the average score
        scores = []
        for i in range(len(y_true)):
            true_text = y_true.iloc[i][self.value]
            pred_text = y_pred.iloc[i][self.value]
            score = self.jaccard(true_text, pred_text)
            scores.append(score)
        
        return sum(scores) / len(scores) if scores else 0.0

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        id_col = "textID"
        # Ensure the id column is of type string
        submission[id_col] = submission[id_col].astype(str)
        ground_truth[id_col] = ground_truth[id_col].astype(str)

        # Sort the submission and ground truth by the id column
        submission = submission.sort_values(by=id_col).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=id_col).reset_index(drop=True)

        # Check if id column values are identical
        if not (submission[id_col].values == ground_truth[id_col].values).all():
            raise InvalidSubmissionError("The values in the id column do not match between submission and ground truth. Please ensure the id values are identical.")

        required_cols = {id_col, self.value}
        submission_cols = set(submission.columns)
        ground_truth_cols = set(ground_truth.columns)

        missing_cols = ground_truth_cols - submission_cols
        extra_cols = submission_cols - ground_truth_cols

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."