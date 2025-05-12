from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class ShopeeProductMatchingMetrics(CompetitionMetrics):
    def __init__(self, value: str = "matches", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        """
        Calculate the mean F1 score for Shopee Product Matching competition.
        
        The mean is calculated in a sample-wise fashion, meaning that an F1 score
        is calculated for every predicted row, then averaged.
        
        Args:
            y_true: DataFrame containing ground truth matches
            y_pred: DataFrame containing predicted matches
            
        Returns:
            float: Mean F1 score across all samples
        """
        # Ensure the posting_id columns are string type
        y_true['posting_id'] = y_true['posting_id'].astype(str)
        y_pred['posting_id'] = y_pred['posting_id'].astype(str)
        
        # Sort by posting_id column which is the first column
        y_true = y_true.sort_values(by='posting_id').reset_index(drop=True)
        y_pred = y_pred.sort_values(by='posting_id').reset_index(drop=True)

        if len(y_true) != len(y_pred):
            raise InvalidSubmissionError("Number of rows in ground truth and submission do not match.")

        f1_scores = []
        # Calculate mean F1 score sample-wise
        for idx in range(len(y_true)):
            # Split space-delimited strings into sets for easier comparisons
            gt_matches = set(y_true.loc[idx, self.value].split())
            pred_matches = set(y_pred.loc[idx, self.value].split())
            
            # Calculate F1 score for the current sample
            if len(pred_matches) == 0 and len(gt_matches) == 0:
                f1 = 1.0
            else:
                common = len(pred_matches.intersection(gt_matches))
                if common == 0:
                    f1 = 0.0
                else:
                    precision = common / len(pred_matches)
                    recall = common / len(gt_matches)
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1)
        
        return float(np.mean(f1_scores))

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        """
        Validate that the submission meets the competition requirements.
        
        Submissions must be a space-delimited list of all posting_ids that match
        the posting in the posting_id column. Posts always self-match.
        Group sizes were capped at 50, so there is no benefit to predict more than 50 matches.
        
        Args:
            submission: DataFrame containing predicted matches
            ground_truth: DataFrame containing ground truth matches
            
        Returns:
            str: Validation message
            
        Raises:
            InvalidSubmissionError: If submission format is invalid
        """
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")
        
        # Ensure posting_id columns are of string type
        submission['posting_id'] = submission['posting_id'].astype(str)
        ground_truth['posting_id'] = ground_truth['posting_id'].astype(str)
        
        # Sort both DataFrames by posting_id to align them
        submission = submission.sort_values(by='posting_id').reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by='posting_id').reset_index(drop=True)

        # Check if posting_id columns match exactly between submission and ground_truth
        if (submission['posting_id'].values != ground_truth['posting_id'].values).any():
            raise InvalidSubmissionError("The posting_id values in submission do not match those in ground truth. Please ensure they are identical.")

        sub_cols = set(submission.columns)
        true_cols = set(ground_truth.columns)

        missing_cols = true_cols - sub_cols
        extra_cols = sub_cols - true_cols

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."