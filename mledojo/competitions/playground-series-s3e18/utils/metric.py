from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError
from sklearn.metrics import roc_auc_score

class PlaygroundSeriesS3E18Metrics(CompetitionMetrics):
    """Metric class for Playground Series S3E18 competition evaluating average ROC AUC for EC1 and EC2 targets."""
    def __init__(self, value: str = "EC1", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Convert the id column to string and sort both dataframes by id
        id_col = y_true.columns[0]
        y_true[id_col] = y_true[id_col].astype(str)
        y_pred[y_pred.columns[0]] = y_pred[y_pred.columns[0]].astype(str)
        
        y_true = y_true.sort_values(by=id_col).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=y_pred.columns[0]).reset_index(drop=True)
        
        target_columns = ["EC1", "EC2"]
        auc_scores = []
        for target in target_columns:
            try:
                # Calculate ROC AUC for the target
                auc = roc_auc_score(y_true[target], y_pred[target])
            except ValueError:
                # In case there is only one class in y_true[target]
                auc = 0.5
            auc_scores.append(auc)
        return np.mean(auc_scores)

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(
                f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows."
            )

        id_col_sub = submission.columns[0]
        id_col_true = ground_truth.columns[0]

        submission[id_col_sub] = submission[id_col_sub].astype(str)
        ground_truth[id_col_true] = ground_truth[id_col_true].astype(str)

        submission = submission.sort_values(by=id_col_sub).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=id_col_true).reset_index(drop=True)

        if not (submission[id_col_sub].values == ground_truth[id_col_true].values).all():
            raise InvalidSubmissionError("ID columns do not match between submission and ground truth. Please ensure the first column values are identical and in the same order.")

        required_columns = {id_col_sub, "EC1", "EC2"}
        submission_columns = set(submission.columns)

        missing_cols = required_columns - submission_columns
        extra_cols = submission_columns - required_columns

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."