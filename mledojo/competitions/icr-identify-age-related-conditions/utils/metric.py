from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class IcrIdentifyAgeRelatedConditionsMetrics(CompetitionMetrics):
    """Metric class for ICR Identify Age Related Conditions competition using Balanced Logarithmic Loss"""
    def __init__(self, value: str = "class_1", higher_is_better: bool = False):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Convert the Id columns to string type
        y_true[y_true.columns[0]] = y_true[y_true.columns[0]].astype(str)
        y_pred[y_pred.columns[0]] = y_pred[y_pred.columns[0]].astype(str)
        
        # Sort by the Id column
        id_col = y_true.columns[0]
        y_true = y_true.sort_values(by=id_col).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=id_col).reset_index(drop=True)
        
        # Convert ground truth Class column to one-hot format
        y_true_onehot = pd.DataFrame()
        y_true_onehot["class_0"] = (y_true["Class"] == 0).astype(int)
        y_true_onehot["class_1"] = (y_true["Class"] == 1).astype(int)
        
        # Rescale predicted probabilities to sum to 1 for each row
        p0 = y_pred["class_0"]
        p1 = y_pred["class_1"]
        row_sum = p0 + p1
        # Avoid division by zero; if row sum is 0, set to a very small number
        row_sum = row_sum.replace(0, 1e-15)
        p0 = p0 / row_sum
        p1 = p1 / row_sum
        
        # Clip predicted probabilities to avoid log(0)
        eps = 1e-15
        p0 = np.clip(p0, eps, 1 - eps)
        p1 = np.clip(p1, eps, 1 - eps)
        
        # Calculate number of observations for each class in ground truth
        N0 = y_true_onehot["class_0"].sum()
        N1 = y_true_onehot["class_1"].sum()
        
        # Compute the balanced logarithmic loss
        loss_class_0 = -(np.sum(y_true_onehot["class_0"] * np.log(p0)) / N0) if N0 > 0 else 0.0
        loss_class_1 = -(np.sum(y_true_onehot["class_1"] * np.log(p1)) / N1) if N1 > 0 else 0.0
        balanced_log_loss = (loss_class_0 + loss_class_1) / 2.0
        
        return balanced_log_loss

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        expected_submission_columns = {"Id", "class_0", "class_1"}
        expected_ground_truth_columns = {"Id", "Class"}
        
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(
                f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows."
            )

        # Convert first column (Id) to string type
        submission[submission.columns[0]] = submission[submission.columns[0]].astype(str)
        ground_truth[ground_truth.columns[0]] = ground_truth[ground_truth.columns[0]].astype(str)
        
        # Sort by the Id column
        id_col = submission.columns[0]
        submission = submission.sort_values(by=id_col).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=ground_truth.columns[0]).reset_index(drop=True)

        # Check if Ids are identical between submission and ground truth
        if not np.array_equal(submission[id_col].values, ground_truth[ground_truth.columns[0]].values):
            raise InvalidSubmissionError("Id column values do not match between submission and ground truth. Please ensure they are identical.")

        submission_cols = set(submission.columns)
        ground_truth_cols = set(ground_truth.columns)

        missing_submission_cols = expected_submission_columns - submission_cols
        extra_submission_cols = submission_cols - expected_submission_columns
        missing_ground_truth_cols = expected_ground_truth_columns - ground_truth_cols
        extra_ground_truth_cols = ground_truth_cols - expected_ground_truth_columns

        if missing_submission_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_submission_cols)}.")
        if extra_submission_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_submission_cols)}.")
        if missing_ground_truth_cols:
            raise InvalidSubmissionError(f"Missing required columns in ground truth: {', '.join(missing_ground_truth_cols)}.")
        if extra_ground_truth_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in ground truth: {', '.join(extra_ground_truth_cols)}.")

        return "Submission is valid."