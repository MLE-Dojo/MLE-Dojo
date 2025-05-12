from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class ChaiiHindiAndTamilQuestionAnsweringMetrics(CompetitionMetrics):
    """
    Metric class for the chaii-hindi-and-tamil-question-answering competition.
    The scoring metric used in this competition is the word-level Jaccard score.
    """
    def __init__(self, value: str = "PredictionString", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def jaccard(self, str1: str, str2: str) -> float:
        a = set(str(str1).lower().split())
        b = set(str(str2).lower().split())
        if not a and not b:
            return 1.0  # if both strings are empty, define similarity as 1
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))
    
    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure the id columns are of string type
        y_true[y_true.columns[0]] = y_true[y_true.columns[0]].astype(str)
        y_pred[y_pred.columns[0]] = y_pred[y_pred.columns[0]].astype(str)
        
        # Sort both dataframes by the id column (first column)
        y_true = y_true.sort_values(by=y_true.columns[0]).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=y_pred.columns[0]).reset_index(drop=True)
        
        # Compute the average word-level Jaccard score
        scores = []
        for gt, pred in zip(y_true[self.value], y_pred[self.value]):
            scores.append(self.jaccard(gt, pred))
        return float(np.mean(scores)) if scores else 0.0

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        
        # Check that the number of rows match
        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")
        
        # Convert id columns to string type and sort by the id column (assumed to be the first column)
        submission[submission.columns[0]] = submission[submission.columns[0]].astype(str)
        ground_truth[ground_truth.columns[0]] = ground_truth[ground_truth.columns[0]].astype(str)
        submission = submission.sort_values(by=submission.columns[0]).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=ground_truth.columns[0]).reset_index(drop=True)
        
        if (submission[submission.columns[0]].values != ground_truth[ground_truth.columns[0]].values).any():
            raise InvalidSubmissionError("The id column values do not match between submission and ground truth. Please ensure the id values are identical and in the same order.")
        
        # Check for required columns in submission and ground truth
        submission_required = {submission.columns[0], self.value}
        ground_truth_required = {ground_truth.columns[0], self.value}
        
        missing_sub = ground_truth_required - set(submission.columns)
        if missing_sub:
            raise InvalidSubmissionError(f"Submission is missing required columns: {', '.join(missing_sub)}.")
        
        extra_sub = set(submission.columns) - submission_required
        if extra_sub:
            raise InvalidSubmissionError(f"Submission contains unexpected extra columns: {', '.join(extra_sub)}.")
        
        return "Submission is valid."