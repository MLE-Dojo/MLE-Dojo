from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class PlaygroundSeriesS3E13Metrics(CompetitionMetrics):
    """Metric class for Kaggle Playground Series S3E13 competition using MPA@3.
    For each test instance, up to 3 predictions are provided (separated by spaces). 
    The earlier the correct prediction appears in the list, the higher the score. 
    The score for an instance is defined as 1/(position of correct prediction) if present, otherwise 0.
    The final score is the mean score across all instances.
    """
    
    def __init__(self, value: str = "prognosis", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Convert the id column (first column) to string type for consistency
        id_col_true = y_true.columns[0]
        id_col_pred = y_pred.columns[0]
        y_true[id_col_true] = y_true[id_col_true].astype(str)
        y_pred[id_col_pred] = y_pred[id_col_pred].astype(str)
        
        # Sort both y_true and y_pred by the id column
        y_true = y_true.sort_values(by=id_col_true).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=id_col_pred).reset_index(drop=True)
        
        # Compute the score for each row using Reciprocal Rank up to 3 predictions.
        scores = []
        for i in range(len(y_true)):
            # Extract ground truth prognosis; ensure it's a string
            true_value = str(y_true[self.value].iloc[i]).strip()
            # Extract predictions; predictions are separated by spaces
            preds = str(y_pred[self.value].iloc[i]).strip().split()
            score = 0.0
            # Check each prediction in order, up to 3
            for rank, pred in enumerate(preds[:3]):
                if pred == true_value:
                    score = 1.0 / (rank + 1)
                    break
            scores.append(score)
        return np.mean(scores)

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Convert the id column (first column) to string type
        id_col_submission = submission.columns[0]
        id_col_ground_truth = ground_truth.columns[0]
        submission[id_col_submission] = submission[id_col_submission].astype(str)
        ground_truth[id_col_ground_truth] = ground_truth[id_col_ground_truth].astype(str)
        
        # Sort both submission and ground truth by the id column
        submission = submission.sort_values(by=id_col_submission).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=id_col_ground_truth).reset_index(drop=True)
        
        # Check if the id columns are identical
        if not (submission[id_col_submission].values == ground_truth[id_col_ground_truth].values).all():
            raise InvalidSubmissionError("ID column values do not match between submission and ground truth. Please ensure the first column values are identical.")

        submission_columns = set(submission.columns)
        ground_truth_columns = set(ground_truth.columns)

        missing_cols = ground_truth_columns - submission_columns
        extra_cols = submission_columns - ground_truth_columns
            
        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."