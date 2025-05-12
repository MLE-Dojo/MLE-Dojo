from typing import Any
import pandas as pd
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class UwMadisonGiTractImageSegmentationMetrics(CompetitionMetrics):
    """
    Metric class for UW-Madison GI Tract Image Segmentation competition.
    The final score is a weighted combination of the mean Dice coefficient and the normalized 3D Hausdorff distance.
    Specifically, for each image the score is:
        score = 0.4 * Dice + 0.6 * (1 - normalized Hausdorff)
    where the Dice coefficient is computed as:
        Dice = 2 * |A ∩ B| / (|A| + |B|)
    and the normalized Hausdorff distance is computed by normalizing the 3D Hausdorff distance
    between predicted and ground truth masks by the maximum possible distance in a normalized image.
    Note: The image is assumed to have a fixed resolution of 512x512 for decoding the run‐length encoding.
    """
    def __init__(self, value: str = "predicted", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def decode_rle(self, rle_string: Any, H: int, W: int) -> np.ndarray:
        """
        Decode a run-length encoded string into a binary mask of shape (H, W).
        The pixels are numbered from top to bottom, then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.
        """
        if pd.isna(rle_string) or rle_string == "":
            return np.zeros((H, W), dtype=np.uint8)
        mask = np.zeros(H * W, dtype=np.uint8)
        tokens = rle_string.split()
        if len(tokens) % 2 != 0:
            raise ValueError("Invalid RLE format; it must contain pairs of numbers.")
        for i in range(0, len(tokens), 2):
            start = int(tokens[i]) - 1  # Convert 1-based to 0-based indexing
            length = int(tokens[i + 1])
            mask[start:start + length] = 1
        # Reshape using Fortran order since pixels are numbered top-to-bottom then left-to-right
        return mask.reshape((H, W), order='F')

    def compute_dice(self, mask_true: np.ndarray, mask_pred: np.ndarray) -> float:
        """
        Compute the Dice coefficient between two binary masks.
        If both masks are empty, return 0.0.
        """
        sum_true = mask_true.sum()
        sum_pred = mask_pred.sum()
        if sum_true + sum_pred == 0:
            return 0.0
        intersection = np.sum(mask_true & mask_pred)
        return 2.0 * intersection / (sum_true + sum_pred)

    def compute_hausdorff(self, mask_true: np.ndarray, mask_pred: np.ndarray) -> float:
        """
        Compute the normalized Hausdorff distance between two binary masks.
        If both masks are empty, return 0.0.
        If one mask is empty and the other is not, return the worst-case distance of 1.0.
        The coordinates are normalized by the image dimensions so that the maximum distance is bounded.
        """
        coords_true = np.column_stack(np.where(mask_true))
        coords_pred = np.column_stack(np.where(mask_pred))
        if coords_true.shape[0] == 0 and coords_pred.shape[0] == 0:
            return 0.0
        if coords_true.shape[0] == 0 or coords_pred.shape[0] == 0:
            return 1.0

        H, W = mask_true.shape
        # Normalize coordinates by image dimensions
        coords_true_norm = coords_true / np.array([H, W])
        coords_pred_norm = coords_pred / np.array([H, W])
        d1 = directed_hausdorff(coords_true_norm, coords_pred_norm)[0]
        d2 = directed_hausdorff(coords_pred_norm, coords_true_norm)[0]
        hd = max(d1, d2)
        # Maximum possible Euclidean distance in a normalized image is sqrt(1^2 + 1^2) = sqrt(2)
        normalized_hd = hd / np.sqrt(2)
        return min(normalized_hd, 1.0)

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        """
        Evaluate the submissions using a weighted combination of Dice and normalized Hausdorff metrics.
        Both y_true and y_pred DataFrames must contain at least the columns:
            id, class, and predicted.
        The final score is computed by averaging the per-image scores:
            final_score = 0.4 * (mean Dice) + 0.6 * (1 - mean normalized Hausdorff)
        """
        # Convert id columns to string and sort by id
        y_true["id"] = y_true["id"].astype(str)
        y_pred["id"] = y_pred["id"].astype(str)
        y_true_sorted = y_true.sort_values(by="id").reset_index(drop=True)
        y_pred_sorted = y_pred.sort_values(by="id").reset_index(drop=True)

        # Assume fixed image dimensions of 512x512
        H, W = 512, 512
        dice_list = []
        hausdorff_list = []

        for idx in range(len(y_true_sorted)):
            true_rle = y_true_sorted.iloc[idx][self.value]
            pred_rle = y_pred_sorted.iloc[idx][self.value]
            mask_true = self.decode_rle(true_rle, H, W)
            mask_pred = self.decode_rle(pred_rle, H, W)
            dice = self.compute_dice(mask_true, mask_pred)
            hd = self.compute_hausdorff(mask_true, mask_pred)
            dice_list.append(dice)
            hausdorff_list.append(hd)

        mean_dice = np.mean(dice_list)
        mean_hd = np.mean(hausdorff_list)
        # Combine metrics: higher Dice and lower Hausdorff are better.
        final_score = 0.4 * mean_dice + 0.6 * (1 - mean_hd)
        return final_score

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        """
        Validate the format of the submission DataFrame against the ground truth.
        The submission must have the columns: id, class, predicted.
        The id column values in the submission must match those in the ground truth after sorting.
        """
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        required_columns = {"id", "class", "predicted"}
        submission_cols = set(submission.columns)
        ground_truth_cols = set(ground_truth.columns)

        missing_cols = required_columns - submission_cols
        extra_cols = submission_cols - ground_truth_cols

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        submission["id"] = submission["id"].astype(str)
        ground_truth["id"] = ground_truth["id"].astype(str)
        submission_sorted = submission.sort_values(by="id").reset_index(drop=True)
        ground_truth_sorted = ground_truth.sort_values(by="id").reset_index(drop=True)

        if not (submission_sorted["id"].values == ground_truth_sorted["id"].values).all():
            raise InvalidSubmissionError("ID column values do not match between submission and ground truth. Please ensure they are identical.")

        return "Submission is valid."