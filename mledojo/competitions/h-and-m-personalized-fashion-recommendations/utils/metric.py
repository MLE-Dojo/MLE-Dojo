from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class HandMPersonalizedFashionRecommendationsMetrics(CompetitionMetrics):
    """Metric class for H&M Personalized Fashion Recommendations competition using Mean Average Precision @ 12."""
    def __init__(self, value: str = "prediction", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure customer_id is of string type
        y_true[y_true.columns[0]] = y_true[y_true.columns[0]].astype(str)
        y_pred[y_pred.columns[0]] = y_pred[y_pred.columns[0]].astype(str)
        # Sort by customer_id column (first column)
        y_true = y_true.sort_values(by=y_true.columns[0]).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=y_pred.columns[0]).reset_index(drop=True)
    
        # Helper function to compute AP@k for a single customer
        def average_precision_at_k(true_items, pred_items, k: int = 12) -> float:
            if not true_items:
                return 0.0
            pred_items = pred_items[:k]
            score = 0.0
            num_hits = 0.0
            for i, p in enumerate(pred_items):
                if p in true_items:
                    num_hits += 1.0
                    score += num_hits / (i + 1.0)
            return score / min(len(true_items), k)
    
        # Merge y_true and y_pred on customer_id
        merged = pd.merge(y_true, y_pred, on=y_true.columns[0], suffixes=('_true', '_pred'))
    
        # Calculate MAP@12 across all customers in the merged dataframe
        ap_scores = []
        for _, row in merged.iterrows():
            # Split the ground truth and predicted strings into lists based on whitespace
            true_list = str(row[f"{self.value}_true"]).split()
            pred_list = str(row[f"{self.value}_pred"]).split()
            ap = average_precision_at_k(true_list, pred_list, k=12)
            ap_scores.append(ap)
    
        # Return the mean of all average precision scores
        return np.mean(ap_scores)

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(
                f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows."
            )

        # Convert customer_id to string type
        submission[submission.columns[0]] = submission[submission.columns[0]].astype(str)
        ground_truth[ground_truth.columns[0]] = ground_truth[ground_truth.columns[0]].astype(str)
        # Sort both submission and ground truth by customer_id (first column)
        submission = submission.sort_values(by=submission.columns[0]).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=ground_truth.columns[0]).reset_index(drop=True)

        # Check if the customer_id values are identical between submission and ground truth
        if not np.array_equal(submission[submission.columns[0]].values, ground_truth[ground_truth.columns[0]].values):
            raise InvalidSubmissionError("The customer_id values do not match between submission and ground truth. Please ensure they are identical.")

        required_columns = {submission.columns[0], self.value}
        submission_columns = set(submission.columns)
        ground_truth_columns = set(ground_truth.columns)

        missing_cols = ground_truth_columns - submission_columns
        extra_cols = submission_columns - ground_truth_columns

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."