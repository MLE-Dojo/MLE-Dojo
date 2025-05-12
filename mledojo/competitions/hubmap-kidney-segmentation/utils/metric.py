from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class HubmapKidneySegmentationMetrics(CompetitionMetrics):
    """Metric class for hubmap-kidney-segmentation competition using mean Dice coefficient."""
    def __init__(self, value: str = "predicted", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def decode_rle(self, rle: str) -> set:
        """Decode a run-length encoded string into a set of pixel indices."""
        if pd.isna(rle) or rle == "":
            return set()
        splits = rle.strip().split()
        if len(splits) % 2 != 0:
            raise ValueError("RLE string has an odd number of elements.")
        nums = list(map(int, splits))
        pixels = set()
        for i in range(0, len(nums), 2):
            start = nums[i]
            length = nums[i + 1]
            pixels.update(range(start, start + length))
        return pixels

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Convert id column to string type
        y_true["id"] = y_true["id"].astype(str)
        y_pred["id"] = y_pred["id"].astype(str)
        # Sort by id
        y_true = y_true.sort_values(by="id").reset_index(drop=True)
        y_pred = y_pred.sort_values(by="id").reset_index(drop=True)
        
        dice_scores = []
        for _, (true_row, pred_row) in enumerate(zip(y_true.itertuples(), y_pred.itertuples())):
            rle_true = true_row.predicted
            rle_pred = pred_row.predicted
            set_true = self.decode_rle(rle_true)
            set_pred = self.decode_rle(rle_pred)
            if len(set_true) == 0 and len(set_pred) == 0:
                dice = 1.0
            else:
                dice = (2 * len(set_true.intersection(set_pred))) / (len(set_true) + len(set_pred))
            dice_scores.append(dice)
        return np.mean(dice_scores)

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Ensure 'id' column is string type
        submission["id"] = submission["id"].astype(str)
        ground_truth["id"] = ground_truth["id"].astype(str)
        # Sort by the 'id' column
        submission = submission.sort_values(by="id").reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by="id").reset_index(drop=True)

        # Check if 'id' columns are identical
        if (submission["id"].values != ground_truth["id"].values).any():
            raise InvalidSubmissionError("Image IDs do not match between submission and ground truth. Please ensure the 'id' column values are identical.")

        required_columns = {"id", "predicted"}
        sub_columns = set(submission.columns)
        missing_cols = required_columns - sub_columns
        # Optional: Remove check for extra columns if they are allowed
        # extra_cols = sub_columns - required_columns

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        # Optional: Remove check for extra columns if they are allowed
        # if extra_cols:
        #     raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        # # Removed check for empty RLE strings as they should be valid (handled by evaluate)
        # empty_predictions = submission[submission["predicted"].isna() | (submission["predicted"] == "")]
        # if not empty_predictions.empty:
        #     raise InvalidSubmissionError(f"Found {len(empty_predictions)} empty predictions. All predictions must contain valid RLE strings.")

        # Validate RLE format for non-empty strings
        for idx, row in submission.iterrows():
            rle = row["predicted"]
            # Check only if rle is not NaN and not an empty string
            if pd.notna(rle) and rle != "":
                splits = rle.strip().split()
                if len(splits) % 2 != 0:
                    raise InvalidSubmissionError(f"Invalid RLE format at row {idx}: '{rle}'. RLE must have an even number of elements.")
                try:
                    nums = list(map(int, splits))
                    # RLE values are pixel start and length, should be non-negative. Start can be 0, length must be >= 1?
                    # The original code checked > 0. Let's stick to that for now, but it might need refinement based on exact RLE definition.
                    # Update: The decode_rle uses range(start, start + length), so start can be 0, length >= 1.
                    # Let's adjust the check slightly: start >= 0, length >= 1.
                    # If start is nums[i] and length is nums[i+1]
                    for i in range(0, len(nums), 2):
                        start = nums[i]
                        length = nums[i+1]
                        if start < 0:
                             raise InvalidSubmissionError(f"Invalid RLE values at row {idx}: '{rle}'. Start index ({start}) cannot be negative.")
                        if length <= 0:
                             raise InvalidSubmissionError(f"Invalid RLE values at row {idx}: '{rle}'. Run length ({length}) must be positive.")

                    # Original check was: if any(num <= 0 for num in nums):
                    # This might be too strict if start index 0 is allowed.
                    # Let's keep the improved check above.

                except ValueError:
                    raise InvalidSubmissionError(f"Invalid RLE format at row {idx}: '{rle}'. RLE must contain only integers.")
                # Ensure RLE values make sense (e.g., don't reference pixels outside image bounds)
                # This would require image dimensions, which are not available here.
                # This level of validation might be better suited elsewhere or skipped if too complex here.


        return "Submission is valid."