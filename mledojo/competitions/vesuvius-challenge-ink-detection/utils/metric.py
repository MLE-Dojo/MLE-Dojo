from typing import Any, Set
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class VesuviusChallengeInkDetectionMetrics(CompetitionMetrics):
    """Metric class for the Vesuvius Challenge Ink Detection competition using a modified F0.5 score."""
    def __init__(self, value: str = "Predicted", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value
        self.beta = 0.5

    def run_length_decode(self, rle: str) -> Set[int]:
        """
        Decodes a run-length encoding string into a set of pixel indices.
        Assumes a space delimited string of even number of positive integers.
        For example, '1 3 10 5' decodes to pixels {1,2,3,10,11,12,13,14}.
        """
        decoded = set()
        if pd.isna(rle) or rle.strip() == "":
            return decoded
        try:
            tokens = rle.split()
            if len(tokens) % 2 != 0:
                raise ValueError("RLE string does not contain an even number of elements.")
            pairs = [int(x) for x in tokens]
        except Exception as e:
            raise ValueError(f"Invalid RLE format: {rle}. Error: {str(e)}")
        for i in range(0, len(pairs), 2):
            start = pairs[i]
            length = pairs[i+1]
            if start < 1 or length < 1:
                raise ValueError(f"Invalid values in RLE: start {start} and length {length} must be positive.")
            # Add pixels from start to start + length - 1
            for pix in range(start, start + length):
                if pix in decoded:
                    raise ValueError("Decoded pixels are duplicated.")
                decoded.add(pix)
        return decoded

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure first column (Id) is string
        y_true[y_true.columns[0]] = y_true[y_true.columns[0]].astype(str)
        y_pred[y_pred.columns[0]] = y_pred[y_pred.columns[0]].astype(str)
        # Sort by Id
        y_true = y_true.sort_values(by=y_true.columns[0]).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=y_pred.columns[0]).reset_index(drop=True)

        scores = []
        # Iterate through each row and compute F0.5 for run-length decoded masks.
        for idx, row in y_true.iterrows():
            # Get ground truth and prediction run-length strings
            true_rle = row[self.value]
            pred_rle = y_pred.loc[idx, self.value]
            
            try:
                true_pixels = self.run_length_decode(true_rle)
            except ValueError as e:
                raise InvalidSubmissionError(f"Error decoding ground truth RLE for Id {row[y_true.columns[0]]}: {str(e)}")
            try:
                pred_pixels = self.run_length_decode(pred_rle)
            except ValueError as e:
                raise InvalidSubmissionError(f"Error decoding submission RLE for Id {row[y_true.columns[0]]}: {str(e)}")
            
            tp = len(true_pixels.intersection(pred_pixels))
            fp = len(pred_pixels - true_pixels)
            fn = len(true_pixels - pred_pixels)
            
            precision_den = tp + fp
            recall_den = tp + fn
            precision = tp / precision_den if precision_den > 0 else 1.0
            recall = tp / recall_den if recall_den > 0 else 1.0
            
            beta_sq = self.beta ** 2
            denom = (beta_sq * precision + recall)
            score = ((1 + beta_sq) * precision * recall / denom) if denom > 0 else 0.0
            scores.append(score)
        
        # Return the average F0.5 score over all rows
        return np.mean(scores)

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(
                f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows."
            )

        # Convert the first column (Id) to string type
        submission[submission.columns[0]] = submission[submission.columns[0]].astype(str)
        ground_truth[ground_truth.columns[0]] = ground_truth[ground_truth.columns[0]].astype(str)
        # Sort by Id
        submission = submission.sort_values(by=submission.columns[0]).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=ground_truth.columns[0]).reset_index(drop=True)

        # Check if first column (Id) values are identical
        if (submission[submission.columns[0]].values != ground_truth[ground_truth.columns[0]].values).any():
            raise InvalidSubmissionError("First column values do not match between submission and ground truth. Please ensure the first column values are identical.")

        sub_cols = set(submission.columns)
        true_cols = set(ground_truth.columns)

        missing_cols = true_cols - sub_cols
        extra_cols = sub_cols - true_cols

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")
        
        # # Validate the run-length encoding format for each submission entry in the designated column.
        # for index, rle in submission[self.value].iteritems():
        #     if not (pd.isna(rle) or isinstance(rle, str)):
        #         raise InvalidSubmissionError(f"RLE value at row {index} must be a string.")

        #     # Attempt to decode to ensure formatting is correct.
        #     try:
        #         self.run_length_decode(rle)
        #     except ValueError as e:
        #         raise InvalidSubmissionError(f"Invalid RLE format in submission at row {index}: {str(e)}")
                
        return "Submission is valid."