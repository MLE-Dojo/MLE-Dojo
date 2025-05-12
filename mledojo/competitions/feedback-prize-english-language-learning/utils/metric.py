from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class FeedbackPrizeEnglishLanguageLearningMetrics(CompetitionMetrics):
    """
    Metric class for Feedback Prize English Language Learning competition using Mean Columnwise Root Mean Squared Error (MCRMSE).
    Lower MCRMSE indicates better performance.
    """
    def __init__(self, value: list = None, higher_is_better: bool = False):
        """
        Initializes the metric.
        :param value: A list of target column names to evaluate.
                      Defaults to ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions'].
        :param higher_is_better: Boolean indicating if a higher score is better (False for error metrics).
        """
        super().__init__(higher_is_better)
        if value is None:
            self.value = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
        else:
            self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        """
        Evaluate the MCRMSE score for the given ground truth and predictions.
        Both dataframes must have the same 'text_id' in the first column and target columns.
        """
        # Ensure text_id column is string type
        y_true[y_true.columns[0]] = y_true[y_true.columns[0]].astype(str)
        y_pred[y_pred.columns[0]] = y_pred[y_pred.columns[0]].astype(str)
        
        # Sort by text_id column
        sort_column = y_true.columns[0]  # Assumes the first column is text_id
        y_true = y_true.sort_values(by=sort_column).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=sort_column).reset_index(drop=True)

        rmses = []
        # Calculate RMSE for each target column
        for col in self.value:
            if col not in y_true.columns or col not in y_pred.columns:
                raise KeyError(f"Column '{col}' is missing from either the ground truth or prediction DataFrame.")
            mse = np.mean(np.square(y_true[col] - y_pred[col]))
            rmse = np.sqrt(mse)
            rmses.append(rmse)
            
        # Calculate mean columnwise RMSE
        mcrmse = np.mean(rmses)
        return mcrmse

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        """
        Validates the submission DataFrame against the ground truth DataFrame.
        Both should be pandas DataFrames with matching 'text_id' and target columns.
        """
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(
                f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows."
            )

        # Convert text_id column to string type
        id_column = submission.columns[0]  # Assumes the first column is text_id
        submission[id_column] = submission[id_column].astype(str)
        ground_truth[id_column] = ground_truth[id_column].astype(str)

        # Sort both by text_id column
        submission = submission.sort_values(by=id_column).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=id_column).reset_index(drop=True)

        # Check if text_id columns are identical
        if not (submission[id_column].values == ground_truth[id_column].values).all():
            raise InvalidSubmissionError("The text_id values do not match between submission and ground truth. Please ensure the text_id column is identical.")

        # Define the required columns
        required_columns = set([id_column] + self.value)
        sub_columns = set(submission.columns)
        gt_columns = set(ground_truth.columns)

        missing_cols = required_columns - sub_columns
        extra_cols = sub_columns - required_columns

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."