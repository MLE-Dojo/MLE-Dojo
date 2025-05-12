from typing import Any
import numpy as np
import pandas as pd
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError
from sklearn.metrics import roc_curve

class Alaska2ImageSteganalysisMetrics(CompetitionMetrics):
    """
    Metric class for the ALASKA2 Image Steganalysis competition using a weighted AUC.
    The weighted AUC is computed by weighting the area under the ROC curve based on the following:
      - For TPR values between 0.0 and 0.4, the corresponding FPR values are weighted by 2.
      - For TPR values between 0.4 and 1.0, the corresponding FPR values are weighted by 1.
    The final weighted AUC score is normalized so that a perfect classifier scores 1 and a random classifier
    obtains a lower score.
    """
    
    def __init__(self, value: str = "Label", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value  # column name used for calculating the score

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure the identifier columns (first column) are strings and sort both dataframes by them
        id_col_true = y_true.columns[0]
        id_col_pred = y_pred.columns[0]
        y_true[id_col_true] = y_true[id_col_true].astype(str)
        y_pred[id_col_pred] = y_pred[id_col_pred].astype(str)
        y_true = y_true.sort_values(by=id_col_true).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=id_col_pred).reset_index(drop=True)
        
        # Extract ground truth labels and prediction scores
        # Higher scores indicate a higher likelihood of hidden message
        y_true_labels = y_true[self.value].values
        y_pred_scores = y_pred[self.value].values

        # Compute ROC curve: fpr, tpr and thresholds
        fpr, tpr, _ = roc_curve(y_true_labels, y_pred_scores)
        
        # Define the TPR thresholds and weights as specified in the competition
        tpr_thresholds = [0.0, 0.4, 1.0]
        weights = [2, 1]
        
        # Calculate weighted AUC
        weighted_auc = 0
        total_weight = sum(weights)
        
        for i in range(len(weights)):
            # Find indices for the current TPR range
            start_idx = np.searchsorted(tpr, tpr_thresholds[i], side='left')
            end_idx = np.searchsorted(tpr, tpr_thresholds[i+1], side='right')
            
            # If we have points in this range, calculate the area
            if start_idx < end_idx:
                # Extract the TPR and FPR values for this range
                segment_tpr = np.concatenate(([tpr_thresholds[i]], tpr[start_idx:end_idx], [tpr_thresholds[i+1]]))
                segment_fpr = np.concatenate(([fpr[start_idx]], fpr[start_idx:end_idx], [fpr[end_idx-1]]))
                
                # Calculate the area using trapezoidal rule and apply weight
                area = np.trapz(segment_fpr, segment_tpr)
                weighted_auc += weights[i] * area
        
        # Normalize by the sum of weights
        weighted_auc = weighted_auc / total_weight
        
        # Invert the score (1 - weighted_auc) since lower FPR is better
        return 1 - weighted_auc

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(
                f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows."
            )

        # Convert the ID column (first column) to string type for both submission and ground truth
        id_col_sub = submission.columns[0]
        id_col_true = ground_truth.columns[0]
        submission[id_col_sub] = submission[id_col_sub].astype(str)
        ground_truth[id_col_true] = ground_truth[id_col_true].astype(str)
        
        # Sort the submission and ground truth by the identifier column
        submission = submission.sort_values(by=id_col_sub).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=id_col_true).reset_index(drop=True)
        
        # Check if identifier columns match exactly
        if not np.array_equal(submission[id_col_sub].values, ground_truth[id_col_true].values):
            raise InvalidSubmissionError("Identifier column values do not match between submission and ground truth. Please ensure the first column values are identical.")

        sub_cols = set(submission.columns)
        true_cols = set(ground_truth.columns)

        missing_cols = true_cols - sub_cols
        extra_cols = sub_cols - true_cols

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."