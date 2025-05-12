from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class TheNatureConservancyFisheriesMonitoringMetrics(CompetitionMetrics):
    """
    Metric class for The Nature Conservancy Fisheries Monitoring competition using multi-class Logarithmic Loss.
    Lower log loss indicates better performance.
    """
    def __init__(self, value: str = None, higher_is_better: bool = False):
        super().__init__(higher_is_better)
        self.value = value
        # Define the expected probability columns (classes) as provided in the competition
        self.classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Convert the image identifiers to string type
        image_col_true = y_true.columns[0]
        image_col_pred = y_pred.columns[0]
        y_true[image_col_true] = y_true[image_col_true].astype(str)
        y_pred[image_col_pred] = y_pred[image_col_pred].astype(str)

        # Sort both DataFrames by the image identifier column
        y_true = y_true.sort_values(by=image_col_true).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=image_col_pred).reset_index(drop=True)

        # Ensure that the image columns match exactly
        if not (y_true[image_col_true].values == y_pred[image_col_pred].values).all():
            raise InvalidSubmissionError("The image identifiers in submission and ground truth do not match.")

        # Extract predicted probabilities (all columns besides the image column)
        probs = y_pred[self.classes].copy()
        
        # Row-normalize predicted probabilities (if row sum is not 1)
        row_sum = probs.sum(axis=1)
        # To avoid division by zero, replace zeros with ones (if any)
        row_sum = row_sum.replace(0, 1)
        probs = probs.divide(row_sum, axis=0)
        
        # Clip probabilities to avoid log(0) issues
        epsilon = 1e-15
        probs = probs.clip(lower=epsilon, upper=1 - epsilon)

        # Calculate log loss
        n = len(y_true)
        log_loss = 0.0
        
        # Debug: Print sample values to understand the issue
        # print(f"Sample true values: {y_true.iloc[0][self.classes]}")
        # print(f"Sample pred values: {probs.iloc[0]}")
        
        for i in range(n):
            for j, cls in enumerate(self.classes):
                # Convert ground truth values to numeric and ensure they are 0 or 1
                # The issue might be that y_true values are strings "0" and "1" instead of numeric
                y_ij = float(y_true.iloc[i][cls]) if isinstance(y_true.iloc[i][cls], str) else y_true.iloc[i][cls]
                p_ij = probs.iloc[i][cls]
                
                # Ensure y_ij is either 0 or 1
                if y_ij not in [0, 1]:
                    raise ValueError(f"Ground truth values must be 0 or 1, got {y_ij} for class {cls} at row {i}")
                
                # Only add to log loss if y_ij is 1 (the true class)
                if y_ij == 1:
                    log_loss -= np.log(p_ij)
        
        log_loss /= n
        return log_loss

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        # Check that submission and ground_truth are pandas DataFrames
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        image_col_sub = submission.columns[0]
        image_col_truth = ground_truth.columns[0]
        submission[image_col_sub] = submission[image_col_sub].astype(str)
        ground_truth[image_col_truth] = ground_truth[image_col_truth].astype(str)

        submission = submission.sort_values(by=image_col_sub).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=image_col_truth).reset_index(drop=True)

        if not (submission[image_col_sub].values == ground_truth[image_col_truth].values).all():
            raise InvalidSubmissionError("The image identifiers in submission and ground truth do not match. Please ensure the first column values are identical.")

        # Validate that submission contains exactly the expected columns.
        expected_submission_cols = set(["image"] + self.classes)
        submission_cols = set(submission.columns)
        if submission_cols != expected_submission_cols:
            missing_cols = expected_submission_cols - submission_cols
            extra_cols = submission_cols - expected_submission_cols
            error_msg = ""
            if missing_cols:
                error_msg += f"Missing required columns: {', '.join(missing_cols)}. "
            if extra_cols:
                error_msg += f"Extra unexpected columns found: {', '.join(extra_cols)}."
            raise InvalidSubmissionError(error_msg.strip())
            
        # Validate that all probability values are between 0 and 1
        for cls in self.classes:
            if (submission[cls] < 0).any() or (submission[cls] > 1).any():
                raise InvalidSubmissionError(f"Probability values for class {cls} must be between 0 and 1.")
            
        return "Submission is valid."