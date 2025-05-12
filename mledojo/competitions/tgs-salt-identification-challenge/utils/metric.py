from typing import Any
import numpy as np
import pandas as pd
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class TgsSaltIdentificationChallengeMetrics(CompetitionMetrics):
    def __init__(self, value: str = "rle_mask", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value
        # Define the fixed image dimensions
        self.img_shape = (101, 101)
        # Define the IoU thresholds as per competition
        self.thresholds = np.arange(0.5, 1.0, 0.05)

    def decode_rle(self, rle_string: str) -> np.ndarray:
        """
        Decode a run-length encoded string into a binary mask.
        The mask is expected to be in Fortran order (column-major) as per competition.
        """
        mask = np.zeros(self.img_shape[0] * self.img_shape[1], dtype=np.uint8)
        if pd.isnull(rle_string) or rle_string == "":
            return mask.reshape(self.img_shape, order='F')
        s = rle_string.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
        starts -= 1  # converting to zero-index
        for start, length in zip(starts, lengths):
            mask[start:start + length] = 1
        return mask.reshape(self.img_shape, order='F')

    def compute_iou(self, pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
        """
        Compute Intersection over Union (IoU) between predicted and true masks.
        Handle the case when both masks are empty.
        """
        intersection = np.logical_and(pred_mask, true_mask).sum()
        union = np.logical_or(pred_mask, true_mask).sum()
        if union == 0:
            return 1.0
        return intersection / union

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure the id column (first column) is treated as string
        y_true[y_true.columns[0]] = y_true[y_true.columns[0]].astype(str)
        y_pred[y_pred.columns[0]] = y_pred[y_pred.columns[0]].astype(str)
        # Sort both DataFrames by the id column
        id_col = y_true.columns[0]
        y_true = y_true.sort_values(by=id_col).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=id_col).reset_index(drop=True)

        ap_scores = []  # list to store average precision per image

        for idx, row in y_true.iterrows():
            true_rle = row[self.value]
            pred_rle = y_pred.loc[idx, self.value]
            true_mask = self.decode_rle(true_rle)
            pred_mask = self.decode_rle(pred_rle)
            iou = self.compute_iou(pred_mask, true_mask)

            precisions = []
            for t in self.thresholds:
                if iou >= t:
                    precisions.append(1.0)
                else:
                    precisions.append(0.0)
            ap = np.mean(precisions)
            ap_scores.append(ap)
        # The final score is the mean average precision over all images
        return float(np.mean(ap_scores))

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(
                f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows."
            )

        # Convert the identifier columns to string type
        id_col = ground_truth.columns[0]
        submission[id_col] = submission[id_col].astype(str)
        ground_truth[id_col] = ground_truth[id_col].astype(str)
        # Sort both DataFrames by the identifier column
        submission = submission.sort_values(by=id_col).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=id_col).reset_index(drop=True)

        # Check if the identifier columns are identical
        if not np.array_equal(submission[id_col].values, ground_truth[id_col].values):
            raise InvalidSubmissionError("Identifier column values do not match between submission and ground truth. Please ensure the first column values are identical.")

        submission_cols = set(submission.columns)
        ground_truth_cols = set(ground_truth.columns)

        missing_cols = ground_truth_cols - submission_cols
        extra_cols = submission_cols - ground_truth_cols

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."