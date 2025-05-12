from typing import Any
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class FlavoursOfPhysicsKernelsOnlyMetrics(CompetitionMetrics):
    """
    Metric class for the flavours-of-physics-kernels-only competition using a
    Weighted Area Under the ROC Curve metric.
    
    The evaluation is computed only on events with min_ANNmuon > 0.4.
    The ROC curve is segmented based on the True Positive Rate (TPR):
      - TPR in [0.0, 0.2] weighted by 2.0
      - TPR in [0.2, 0.4] weighted by 1.5
      - TPR in [0.4, 0.6] weighted by 1.0
      - TPR in [0.6, 0.8] weighted by 0.5
      - TPR above 0.8 has weight 0
  
    The final score is normalized so that a perfect classifier gets a score of 1.0,
    and a random classifier gets a score of 0.0.
    """
    def __init__(self, value: str = "prediction", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        # 'value' here refers to the submission column used for scoring.
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        """
        Evaluate the Weighted AUC score.
        Assumes y_true contains columns: [id, prediction, min_ANNmuon]
        and y_pred contains columns: [id, prediction].
        The evaluation is done only on rows where min_ANNmuon > 0.4.
        Note: For ground truth, the 'prediction' column is the same as the 'signal' column.
        """
        # Ensure id columns are treated as strings and sort by id
        y_true["id"] = y_true["id"].astype(str)
        y_pred["id"] = y_pred["id"].astype(str)
        
        y_true = y_true.sort_values(by="id").reset_index(drop=True)
        y_pred = y_pred.sort_values(by="id").reset_index(drop=True)
        
        # Merge the ground truth and predictions on the id column (align rows)
        merged = pd.merge(y_true, y_pred, on="id", suffixes=("_true", "_pred"))
        
        # Filter only events with min_ANNmuon > 0.4
        if "min_ANNmuon" not in merged.columns:
            raise InvalidSubmissionError("Ground truth data must contain 'min_ANNmuon' column for evaluation.")
        merged = merged[merged["min_ANNmuon"] > 0.4]
        if merged.empty:
            raise InvalidSubmissionError("No events with min_ANNmuon > 0.4 were found for evaluation.")
        
        # Extract true labels and prediction scores.
        # For ground truth, the 'prediction' column is the same as the 'signal' column
        y_true_labels = merged["prediction_true"].values
        y_scores = merged["prediction_pred"].values
        
        # Compute ROC curve: fpr, tpr, thresholds
        fpr, tpr, _ = roc_curve(y_true_labels, y_scores)
        
        # Create a fine grid for TPR between 0 and 1 for integration.
        tpr_grid = np.linspace(0.0, 1.0, num=10000)
        # Interpolate corresponding FPR values over the TPR grid.
        # Since tpr is monotonically increasing but may not start at 0.0 and end at 1.0,
        # we force the interpolation to be defined on the entire [0, 1] with extrapolation.
        fpr_interp = np.interp(tpr_grid, tpr, fpr)
        
        # Define the TPR segments and their corresponding weights.
        segments = [
            (0.0, 0.2, 2.0),
            (0.2, 0.4, 1.5),
            (0.4, 0.6, 1.0),
            (0.6, 0.8, 0.5)
            # TPR above 0.8 has weight 0, so we ignore it.
        ]
        
        weighted_area = 0.0
        weighted_random_area = 0.0
        
        for t_low, t_high, weight in segments:
            # Select indices in the TPR grid within the segment.
            mask = (tpr_grid >= t_low) & (tpr_grid <= t_high)
            tpr_segment = tpr_grid[mask]
            fpr_segment = fpr_interp[mask]
            
            # If the segment is empty because of grid resolution, approximate using endpoints.
            if tpr_segment.size < 2:
                # Get interpolated fpr at boundaries.
                fpr_low = np.interp(t_low, tpr, fpr)
                fpr_high = np.interp(t_high, tpr, fpr)
                partial_area = 0.5 * (fpr_low + fpr_high) * (t_high - t_low)
            else:
                partial_area = np.trapz(fpr_segment, tpr_segment)
            
            weighted_area += weight * partial_area
            # For a random classifier, fpr = tpr so area = integral of tpr dtpr = 0.5*(b^2 - a^2)
            random_area = 0.5 * (t_high**2 - t_low**2)
            weighted_random_area += weight * random_area
        
        # Normalize the weighted area measure so that:
        # - Perfect classifier (fpr = 0 for all tpr) yields score 1.0.
        # - Random classifier yields score 0.0.
        score = 1.0 - (weighted_area / weighted_random_area)
        return score

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        """
        Validate that the submission DataFrame contains the required columns and that
        the id values match those in the ground truth.
        Expected submission columns: ['id', 'prediction']
        Expected ground truth first column: 'id'
        """
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        
        required_submission_cols = {"id", "prediction"}
        submission_cols = set(submission.columns)
        missing_cols = required_submission_cols - submission_cols
        extra_cols = submission_cols - required_submission_cols
        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")
        
        # Convert id columns to string type and sort by id
        submission["id"] = submission["id"].astype(str)
        ground_truth["id"] = ground_truth["id"].astype(str)
        
        submission_sorted = submission.sort_values(by="id").reset_index(drop=True)
        ground_truth_sorted = ground_truth.sort_values(by="id").reset_index(drop=True)
        
        if not np.array_equal(submission_sorted["id"].values, ground_truth_sorted["id"].values):
            raise InvalidSubmissionError("The 'id' column values in submission and ground truth do not match. Please ensure they are identical.")
        
        return "Submission is valid."