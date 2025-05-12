from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class AirbnbRecruitingNewUserBookingsMetrics(CompetitionMetrics):
    """
    Metric class for the Airbnb Recruiting New User Bookings competition using NDCG@5.
    For each user, the ground truth destination country is given relevance 1 while all other predictions
    are given relevance 0. The NDCG is computed per user and averaged across all users.
    """
    def __init__(self, value: str = "country", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, ground_truth: pd.DataFrame, submission: pd.DataFrame) -> float:
        """
        Evaluate the submission using NDCG@5.
        Both ground_truth and submission DataFrames are expected to have two columns: 'id' and 'country'.
        The ground_truth DataFrame should have one row per user indicating the correct destination.
        The submission DataFrame should contain up to 5 predictions per user (ordered with the most confident first).

        The NDCG@5 for a single user is computed as:
          NDCG = 1 / log2(position + 1)  if the correct country is among the predictions,
                 0 otherwise.
        The final score is the average NDCG over all users.
        """
        # Ensure the id columns are strings and sort by 'id'
        ground_truth["id"] = ground_truth["id"].astype(str)
        submission["id"] = submission["id"].astype(str)
        ground_truth = ground_truth.sort_values(by="id").reset_index(drop=True)
        submission = submission.sort_values(by="id").reset_index(drop=True)

        # Create a mapping from user id to true country
        true_dict = ground_truth.set_index("id")[self.value].to_dict()
        # Group submission by user id preserving the order of predictions as they appear
        grouped = submission.groupby("id")[self.value].apply(list)

        ndcg_scores = []
        for user_id, true_country in true_dict.items():
            # Get the list of predicted countries for the user.
            user_predictions = grouped.get(user_id, [])
            # Only consider up to the first 5 predictions
            preds = user_predictions[:5]
            # Compute DCG: if the true country is predicted at position i (0-indexed),
            # then score = 1 / log2(i+2). If not predicted, score = 0.
            try:
                pos = preds.index(true_country)
                ndcg = 1.0 / np.log2(pos + 2)
            except ValueError:
                ndcg = 0.0
            ndcg_scores.append(ndcg)

        # Return the average NDCG@5 score across all users.
        return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        """
        Validate the submission file.
        Both submission and ground_truth must be pandas DataFrames with exactly the columns 'id' and 'country'.
        Additionally, the set of user ids in submission must match the set in ground_truth.
        Each user may have at most 5 predictions.
        """
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        required_cols = {"id", "country"}
        sub_cols = set(submission.columns)
        true_cols = set(ground_truth.columns)

        missing_cols = required_cols - sub_cols
        extra_cols = sub_cols - required_cols
        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        # Convert the id columns to string
        submission["id"] = submission["id"].astype(str)
        ground_truth["id"] = ground_truth["id"].astype(str)

        # Group submission by 'id' and validate the number of predictions per user (maximum 5 allowed)
        grouped = submission.groupby("id").size()
        if (grouped > 5).any():
            problematic_ids = grouped[grouped > 5].index.tolist()
            raise InvalidSubmissionError(f"The following user ids have more than 5 predictions: {', '.join(problematic_ids)}.")

        # Check if every id in ground_truth exists in submission
        submission_ids = set(submission["id"])
        truth_ids = set(ground_truth["id"])
        if submission_ids != truth_ids:
            missing_ids = truth_ids - submission_ids
            extra_ids = submission_ids - truth_ids
            msg = ""
            if missing_ids:
                msg += f"Missing ids in submission: {', '.join(missing_ids)}. "
            if extra_ids:
                msg += f"Extra ids in submission: {', '.join(extra_ids)}."
            raise InvalidSubmissionError(msg.strip())

        return "Submission is valid."