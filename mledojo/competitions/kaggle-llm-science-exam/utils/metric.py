from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class KaggleLlmScienceExamMetrics(CompetitionMetrics):
    """Metric class for kaggle-llm-science-exam competition using Mean Average Precision @ 3 (MAP@3)"""
    def __init__(self, value: str = "answer", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure the id column is string type and sort both dataframes by the id column.
        y_true["id"] = y_true["id"].astype(str)
        y_pred["id"] = y_pred["id"].astype(str)
        y_true = y_true.sort_values(by="id").reset_index(drop=True)
        y_pred = y_pred.sort_values(by="id").reset_index(drop=True)
        
        # Check that the lengths match
        if len(y_true) != len(y_pred):
            raise InvalidSubmissionError("Number of rows in predictions and ground truth do not match.")

        # Retrieve ground truth answers from y_true
        true_answers = y_true[self.value].tolist()
        # Retrieve predictions from y_pred - predictions are space-separated labels
        predictions = y_pred[self.value].tolist()
        
        map3 = 0.0
        n = len(true_answers)
        
        for true_answer, pred_str in zip(true_answers, predictions):
            # Split the predicted string into a list of predictions
            pred_list = pred_str.split()[:3]  # Take only first 3 predictions
            
            # Calculate precision at each position
            score = 0.0
            seen_correct = False
            for k, pred in enumerate(pred_list, start=1):
                # Only count first occurrence of correct answer
                if pred == true_answer and not seen_correct:
                    score = 1.0 / k  # Precision at position k
                    seen_correct = True
                    break
            
            map3 += score
            
        # Return mean average precision
        return map3 / n if n > 0 else 0.0

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        required_submission_cols = {"id", self.value}
        required_truth_cols = {"id", self.value}

        submission_cols = set(submission.columns)
        truth_cols = set(ground_truth.columns)
            
        missing_sub_cols = required_submission_cols - submission_cols
        missing_truth_cols = required_truth_cols - truth_cols

        if missing_sub_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_sub_cols)}.")
        if missing_truth_cols:
            raise InvalidSubmissionError(f"Missing required columns in ground truth: {', '.join(missing_truth_cols)}.")

        # Convert id columns to string type
        submission["id"] = submission["id"].astype(str)
        ground_truth["id"] = ground_truth["id"].astype(str)

        # Sort both dataframes by the id column
        submission = submission.sort_values(by="id").reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by="id").reset_index(drop=True)

        # Check if id columns are identical
        if not np.array_equal(submission["id"].values, ground_truth["id"].values):
            raise InvalidSubmissionError("The 'id' column values do not match between submission and ground truth. Please ensure they are identical.")

        return "Submission is valid."