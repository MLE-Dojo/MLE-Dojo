from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class HotelId2021Fgvc8Metrics(CompetitionMetrics):
    """Metric class for Hotel Recognition to Combat Human Trafficking competition using Mean Average Precision @ 5."""
    def __init__(self, value: str = "hotel_id", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Convert the id columns ("image") to string type
        id_col = y_true.columns[0]
        y_true[id_col] = y_true[id_col].astype(str)
        y_pred[y_pred.columns[0]] = y_pred[y_pred.columns[0]].astype(str)

        # Sort the dataframes by the image column to ensure alignment
        y_true = y_true.sort_values(by=id_col).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=y_pred.columns[0]).reset_index(drop=True)

        # Calculate Mean Average Precision @ 5
        scores = []
        # Loop over each row
        for i in range(len(y_true)):
            true_id = str(y_true.loc[i, self.value])
            preds = str(y_pred.loc[i, self.value]).split()
            # Consider only top 5 predictions
            preds = preds[:5]
            score = 0.0
            # If the true label is in the predictions, average precision becomes 1/(rank)
            if true_id in preds:
                rank = preds.index(true_id) + 1  # ranks are 1-indexed
                score = 1.0 / rank
            scores.append(score)
        return np.mean(scores)

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Convert the id columns ("image") to string type
        submission[submission.columns[0]] = submission[submission.columns[0]].astype(str)
        ground_truth[ground_truth.columns[0]] = ground_truth[ground_truth.columns[0]].astype(str)
        # Sort the submission and ground truth by their id column ("image")
        submission = submission.sort_values(by=submission.columns[0]).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=ground_truth.columns[0]).reset_index(drop=True)

        # Check if the id columns are identical
        if not (submission[submission.columns[0]].values == ground_truth[ground_truth.columns[0]].values).all():
            raise InvalidSubmissionError("The values in the id column do not match between submission and ground truth. Please ensure they are identical.")

        # Check that submission only has 'image' and 'hotel_id' columns
        expected_cols = {"image", "hotel_id"}
        sub_cols = set(submission.columns)
        
        if sub_cols != expected_cols:
            missing_cols = expected_cols - sub_cols
            extra_cols = sub_cols - expected_cols
            
            if missing_cols:
                raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
            if extra_cols:
                raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}. Submission should only contain 'image' and 'hotel_id' columns.")

        return "Submission is valid."