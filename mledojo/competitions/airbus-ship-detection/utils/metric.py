from typing import Any, List
import numpy as np
import pandas as pd
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class AirbusShipDetectionMetrics(CompetitionMetrics):
    """
    Metric class for the Airbus Ship Detection competition.
    The metric calculates the mean F2 Score across a range of IoU thresholds.
    """
    def __init__(self, value: str = "EncodedPixels", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def _decode_rle(self, rle: str, shape: tuple = (768, 768)) -> np.ndarray:
        """
        Decodes a run-length encoded string into a binary mask.
        If rle is empty or NaN, returns an array of zeros.
        """
        mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        if pd.isna(rle) or rle == "":
            return mask.reshape(shape)
        s = rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
        starts -= 1  # Convert 1-indexed to 0-indexed
        for start, length in zip(starts, lengths):
            mask[start: start + length] = 1
        return mask.reshape(shape)

    def _compute_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Computes the Intersection over Union (IoU) between two binary masks.
        """
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if union == 0:
            return 0.0
        return intersection / union

    def _compute_f2_score(self, masks_gt: List[np.ndarray], masks_pred: List[np.ndarray], threshold: float) -> float:
        """
        Computes the F2 score for a given image at a specific IoU threshold.
        The matching is done using a greedy algorithm.
        """
        num_gt = len(masks_gt)
        num_pred = len(masks_pred)
        
        # Special case: no ground truth and no prediction => perfect score.
        if num_gt == 0 and num_pred == 0:
            return 1.0

        # Build IoU matrix
        iou_matrix = np.zeros((num_gt, num_pred), dtype=np.float32)
        for i in range(num_gt):
            for j in range(num_pred):
                iou_matrix[i, j] = self._compute_iou(masks_gt[i], masks_pred[j])
                
        tp = 0
        # Greedy matching: find the highest IoU that exceeds threshold repeatedly
        while True:
            if iou_matrix.size == 0:
                break
            max_iou = iou_matrix.max()
            if max_iou < threshold:
                break
            idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            tp += 1
            # Remove the matched ground truth and prediction by setting their rows and cols to -1.
            iou_matrix[idx[0], :] = -1
            iou_matrix[:, idx[1]] = -1

        fp = num_pred - tp
        fn = num_gt - tp
        
        # F2 score formula: (1+β²)*TP / ((1+β²)*TP + β²*FN + FP) where β=2
        # This simplifies to: 5*TP / (5*TP + 4*FN + FP)
        denominator = (5 * tp + 4 * fn + fp)
        if denominator == 0:
            return 0.0
        return (5 * tp) / denominator

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Convert the image id column (first column) to string and sort both dataframes by it.
        y_true[y_true.columns[0]] = y_true[y_true.columns[0]].astype(str)
        y_pred[y_pred.columns[0]] = y_pred[y_pred.columns[0]].astype(str)
        y_true = y_true.sort_values(by=y_true.columns[0]).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=y_pred.columns[0]).reset_index(drop=True)

        # Group by image id for ground truth and predictions.
        gt_groups = y_true.groupby(y_true.columns[0])[self.value].apply(list).to_dict()
        pred_groups = y_pred.groupby(y_pred.columns[0])[self.value].apply(list).to_dict()

        # Get union of all image ids from both ground truth and submission.
        all_image_ids = set(gt_groups.keys()).union(set(pred_groups.keys()))
        # IoU thresholds from 0.5 to 0.95 with a step of 0.05
        thresholds = np.arange(0.5, 1.0, 0.05)
        image_scores = []

        for image_id in all_image_ids:
            gt_rles = gt_groups.get(image_id, [])
            pred_rles = pred_groups.get(image_id, [])

            # Decode RLEs into masks
            gt_masks = [self._decode_rle(rle) for rle in gt_rles if (not pd.isna(rle) and rle != "")]
            pred_masks = [self._decode_rle(rle) for rle in pred_rles if (not pd.isna(rle) and rle != "")]

            # Check for overlapping masks in predictions
            for i in range(len(pred_masks)):
                for j in range(i + 1, len(pred_masks)):
                    if np.logical_and(pred_masks[i], pred_masks[j]).any():
                        raise InvalidSubmissionError(f"Overlapping ship predictions found for image {image_id}.")

            # Compute F2 score across thresholds for the image.
            f2_scores = []
            for thr in thresholds:
                f2 = self._compute_f2_score(gt_masks, pred_masks, thr)
                f2_scores.append(f2)
            # Average F2 score for this image across all thresholds
            image_score = np.mean(f2_scores)
            image_scores.append(image_score)
            
        # Return the overall score as the mean of image scores.
        return float(np.mean(image_scores))

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        # It's possible to have multiple rows per image for this competition.
        if submission.shape[0] < ground_truth.shape[0]:
            raise InvalidSubmissionError(f"Submission has fewer rows ({submission.shape[0]}) than ground truth ({ground_truth.shape[0]}). Each predicted segmentation should be a separate row.")

        # Convert the image id column (assumed to be the first column) to string.
        submission[submission.columns[0]] = submission[submission.columns[0]].astype(str)
        ground_truth[ground_truth.columns[0]] = ground_truth[ground_truth.columns[0]].astype(str)

        # Sort the submission and ground truth by the first column.
        submission = submission.sort_values(by=submission.columns[0]).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=ground_truth.columns[0]).reset_index(drop=True)

        # Check that the required column exists.
        required_cols = {ground_truth.columns[0], self.value}
        sub_cols = set(submission.columns)
        missing_cols = required_cols - sub_cols
        extra_cols = sub_cols - required_cols

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        # Validate RLE format
        for _, row in submission.iterrows():
            rle = row[self.value]
            if pd.isna(rle) or rle == "":
                continue  # Empty prediction is valid (no ship)
                
            # Check if RLE is properly formatted
            s = str(rle).split()
            if len(s) % 2 != 0:
                raise InvalidSubmissionError(f"Invalid RLE format for image {row[submission.columns[0]]}. RLE must contain pairs of values.")
                
            # Check if pairs are sorted and positive
            starts = [int(s[i]) for i in range(0, len(s), 2)]
            lengths = [int(s[i]) for i in range(1, len(s), 2)]
            
            if any(start <= 0 for start in starts):
                raise InvalidSubmissionError(f"Invalid RLE format for image {row[submission.columns[0]]}. Start positions must be positive.")
                
            if any(length <= 0 for length in lengths):
                raise InvalidSubmissionError(f"Invalid RLE format for image {row[submission.columns[0]]}. Run lengths must be positive.")
                
        # Check for overlapping masks within the same image
        image_groups = submission.groupby(submission.columns[0])
        for image_id, group in image_groups:
            rles = group[self.value].tolist()
            masks = [self._decode_rle(rle) for rle in rles if (not pd.isna(rle) and rle != "")]
            
            for i in range(len(masks)):
                for j in range(i + 1, len(masks)):
                    if np.logical_and(masks[i], masks[j]).any():
                        raise InvalidSubmissionError(f"Overlapping ship predictions found for image {image_id}.")

        return "Submission is valid."