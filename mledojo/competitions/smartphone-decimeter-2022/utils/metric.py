from typing import Any
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, asin
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class SmartphoneDecimeter2022Metrics(CompetitionMetrics):
    """
    Metric class for the Smartphone Decimeter 2022 Competition.
    
    The score is computed as follows:
      1. For each submission row (matched with ground truth via "tripId" and "UnixTimeMillis"),
         compute the horizontal distance error (in meters) between the predicted and true latitude/longitude using the haversine formula.
      2. For each phone, calculate the 50th and 95th percentile of the distance errors.
      3. Average these two percentiles for each phone.
      4. The final score is the mean of these averages across all phones.
    
    Note: Lower score indicates better performance.
    """
    def __init__(self, value: str = "LatitudeDegrees", higher_is_better: bool = False):
        super().__init__(higher_is_better)
        # 'value' is set to the name of one of the prediction columns, but note that evaluation uses both LatitudeDegrees and LongitudeDegrees.
        self.value = value

    def haversine(self, lat1, lon1, lat2, lon2):
        """
        Compute the haversine distance in meters between two points given in decimal degrees.
        """
        # Earth radius in meters
        R = 6371000  
        # Convert decimal degrees to radians 
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1 
        dlon = lon2 - lon1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        return R * c

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure the key identifier columns are of string type for "tripId"
        y_true["tripId"] = y_true["tripId"].astype(str)
        y_pred["tripId"] = y_pred["tripId"].astype(str)
        
        # Sort both dataframes by "tripId" and "UnixTimeMillis"
        y_true = y_true.sort_values(by=["tripId", "UnixTimeMillis"]).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=["tripId", "UnixTimeMillis"]).reset_index(drop=True)
        
        # Merge the two DataFrames on "tripId" and "UnixTimeMillis"
        merged = pd.merge(
            y_true,
            y_pred,
            on=["tripId", "UnixTimeMillis"],
            suffixes=("_true", "_pred")
        )
        
        if merged.empty:
            raise InvalidSubmissionError("Merged DataFrame is empty. Please ensure the submission has matching tripId and UnixTimeMillis values with the ground truth.")

        # Compute the horizontal distance error for each row using the haversine formula
        merged["error"] = merged.apply(
            lambda row: self.haversine(row["LatitudeDegrees_true"], row["LongitudeDegrees_true"],
                                       row["LatitudeDegrees_pred"], row["LongitudeDegrees_pred"]),
            axis=1
        )
        
        # Group by tripId and compute the 50th and 95th percentiles, then average them per phone
        phone_scores = merged.groupby("tripId")["error"].apply(
            lambda errors: (np.percentile(errors, 50) + np.percentile(errors, 95)) / 2.0
        )
        
        # Final score is the mean of these averages across all phones
        final_score = phone_scores.mean()
        return final_score

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        required_columns = {"tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"}
        
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        
        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(
                f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). "
                "Please ensure both have the same number of rows."
            )
        
        submission_cols = set(submission.columns)
        gt_cols = set(ground_truth.columns)
        
        missing_cols = required_columns - submission_cols
        extra_cols = submission_cols - required_columns
        
        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        # Ensure that the "tripId" column is of string type in both DataFrames
        submission["tripId"] = submission["tripId"].astype(str)
        ground_truth["tripId"] = ground_truth["tripId"].astype(str)
        
        # Sort both DataFrames by "tripId" and "UnixTimeMillis"
        submission_sorted = submission.sort_values(by=["tripId", "UnixTimeMillis"]).reset_index(drop=True)
        ground_truth_sorted = ground_truth.sort_values(by=["tripId", "UnixTimeMillis"]).reset_index(drop=True)
        
        # Verify that "tripId" and "UnixTimeMillis" columns match between submission and ground truth
        if not submission_sorted["tripId"].equals(ground_truth_sorted["tripId"]) or not submission_sorted["UnixTimeMillis"].equals(ground_truth_sorted["UnixTimeMillis"]):
            raise InvalidSubmissionError("The 'tripId' and 'UnixTimeMillis' columns in submission do not match those in ground truth. Please ensure they are identical and in the correct order.")

        return "Submission is valid."