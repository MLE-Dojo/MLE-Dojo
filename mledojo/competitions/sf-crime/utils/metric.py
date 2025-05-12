from typing import Any
import pandas as np
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError
from sklearn.metrics import log_loss

class SfCrimeMetrics(CompetitionMetrics):
    """
    Metric class for the sf-crime competition using multi-class logarithmic loss.
    
    The submission file should contain an 'Id' column followed by one column for each crime category.
    The evaluation converts the predicted probabilities by clipping and normalizing each row,
    then computes the log loss based on the one-hot encoded ground truth.
    """
    
    def __init__(self, value: str = "Id", higher_is_better: bool = False):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure the id column is of string type
        id_col = y_true.columns[0]
        y_true[id_col] = y_true[id_col].astype(str)
        y_pred[id_col] = y_pred[id_col].astype(str)
        
        # Sort both dataframes by the id column
        y_true = y_true.sort_values(by=id_col).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=id_col).reset_index(drop=True)
        
        # The first column is the id, the remaining columns are the crime categories.
        category_cols = y_true.columns[1:]
        
        # Extract ground truth as one-hot encoded array.
        # Assuming that in y_true exactly one column among the candidate columns is 1 and the rest are 0.
        true_vals = y_true[category_cols].values
        # Convert one-hot encoding to vector of labels (index of the true category)
        true_labels = np.argmax(true_vals, axis=1)

        # Extract predicted probabilities for the same category columns.
        pred_vals = y_pred[category_cols].values
        
        # Clip predicted probabilities to avoid log(0) issues.
        clipped_preds = np.clip(pred_vals, 1e-15, 1 - 1e-15)
        # Normalize each row so that probabilities sum to 1.
        row_sums = np.sum(clipped_preds, axis=1, keepdims=True)
        norm_preds = clipped_preds / row_sums

        # Compute multi-class logarithmic loss.
        # Note: log_loss computes the average log loss across samples.
        score = log_loss(true_labels, norm_preds, labels=np.arange(len(category_cols)))
        return score

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
    
        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")
    
        # Convert the first column (assumed to be the id) to string type in both dataframes.
        id_col = submission.columns[0]
        submission[id_col] = submission[id_col].astype(str)
        ground_truth[ground_truth.columns[0]] = ground_truth[ground_truth.columns[0]].astype(str)
    
        # Sort both submission and ground truth by the id column.
        submission = submission.sort_values(by=id_col).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=ground_truth.columns[0]).reset_index(drop=True)
    
        # Check if the id values are identical.
        if not (submission[id_col].values == ground_truth[ground_truth.columns[0]].values).all():
            raise InvalidSubmissionError("The id values in the submission do not match those in the ground truth. Please ensure they are identical and in the same order.")
    
        # Ensure submission contains exactly the same columns as ground truth.
        submission_cols = set(submission.columns)
        truth_cols = set(ground_truth.columns)
        missing_cols = truth_cols - submission_cols
        extra_cols = submission_cols - truth_cols
        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")
    
        return "Submission is valid."