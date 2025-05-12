from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError
from sklearn.metrics import roc_auc_score

class HomesiteQuoteConversionMetrics(CompetitionMetrics):
    """Metric class for Homesite Quote Conversion competition using ROC AUC."""
    def __init__(self, value: str = "QuoteConversion_Flag", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure the id column is QuoteNumber and convert it to string for proper sorting
        y_true["QuoteNumber"] = y_true["QuoteNumber"].astype(str)
        y_pred["QuoteNumber"] = y_pred["QuoteNumber"].astype(str)

        # Sort both dataframes by QuoteNumber
        y_true = y_true.sort_values(by="QuoteNumber").reset_index(drop=True)
        y_pred = y_pred.sort_values(by="QuoteNumber").reset_index(drop=True)

        # Calculate the ROC AUC score between ground truth and predictions
        return roc_auc_score(y_true[self.value], y_pred[self.value])

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Check for required columns: QuoteNumber and QuoteConversion_Flag
        required_columns = {"QuoteNumber", "QuoteConversion_Flag"}
        submission_columns = set(submission.columns)
        ground_truth_columns = set(ground_truth.columns)

        missing_cols = required_columns - submission_columns
        extra_cols = submission_columns - required_columns
        
        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        # Convert QuoteNumber to string and sort by it
        submission["QuoteNumber"] = submission["QuoteNumber"].astype(str)
        ground_truth["QuoteNumber"] = ground_truth["QuoteNumber"].astype(str)
        submission = submission.sort_values(by="QuoteNumber").reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by="QuoteNumber").reset_index(drop=True)

        # Check if QuoteNumber values match between submission and ground truth
        if not (submission["QuoteNumber"].values == ground_truth["QuoteNumber"].values).all():
            raise InvalidSubmissionError("QuoteNumber values do not match between submission and ground truth. Please ensure the first column values are identical.")

        return "Submission is valid."