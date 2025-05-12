from typing import Any
import pandas as pd
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError
from sklearn.metrics import accuracy_score

class MultiModalGestureRecognitionMetrics(CompetitionMetrics):
    """
    Metric class for Multi-modal Gesture Recognition competition using Levenshtein distance.
    """
    def __init__(self, value: str = "Sequence", higher_is_better: bool = False):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        from Levenshtein import distance
        import numpy as np
        
        # Prepare data for metric calculation
        answers_sequences, submission_sequences = self._prepare_for_metric(y_pred, y_true)
        
        # Calculate the sum of Levenshtein distances
        sum_of_distances = np.sum([distance(a, b) for a, b in zip(answers_sequences, submission_sequences)])
        total_num_gestures = sum(len(x) for x in answers_sequences)
        
        # Return the final error rate
        return sum_of_distances / total_num_gestures

    def _prepare_for_metric(self, submission: pd.DataFrame, answers: pd.DataFrame):
        """Helper method to prepare dataframes for metric calculation"""
        if len(submission) != len(answers):
            raise InvalidSubmissionError("Submission and answers must have the same length")

        if "Id" not in submission.columns:
            raise InvalidSubmissionError("Submission must have an 'Id' column")

        if "Sequence" not in submission.columns:
            raise InvalidSubmissionError("Submission must have a 'Sequence' column")

        assert "Id" in answers.columns, "Answers must have 'Id' column"
        assert "Sequence" in answers.columns, "Answers must have 'Sequence' column"

        submission = submission.sort_values("Id")
        answers = answers.sort_values("Id")

        if (submission["Id"].values != answers["Id"].values).any():
            raise InvalidSubmissionError("Submission and answers must have the same ids")

        # Read as strings, convert to list of numbers
        submission["Sequence"] = submission["Sequence"].astype(str)
        answers["Sequence"] = answers["Sequence"].astype(str)
        answers_sequences = [list(map(int, x.split())) for x in answers["Sequence"]]
        try:
            submission_sequences = [list(map(int, x.split())) for x in submission["Sequence"]]
        except ValueError as e:
            raise InvalidSubmissionError(
                f"Submission sequences must be integers separated by spaces. Failed to convert Sequence to list of integers: {e}"
            )

        return answers_sequences, submission_sequences

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        # Check for required columns
        if "Id" not in submission.columns:
            raise InvalidSubmissionError("Submission must have an 'Id' column")
        if "Sequence" not in submission.columns:
            raise InvalidSubmissionError("Submission must have a 'Sequence' column")

        # Check row count
        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Convert IDs to string type and sort
        submission["Id"] = submission["Id"].astype(str)
        ground_truth["Id"] = ground_truth["Id"].astype(str)
        submission = submission.sort_values(by="Id").reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by="Id").reset_index(drop=True)

        # Check if IDs match
        if not (submission["Id"].values == ground_truth["Id"].values).all():
            raise InvalidSubmissionError("The sequence IDs do not match between submission and ground truth. Please ensure they are identical.")

        # Validate sequence format
        try:
            for seq in submission["Sequence"]:
                list(map(int, str(seq).split()))
        except ValueError as e:
            raise InvalidSubmissionError(f"Submission sequences must be integers separated by spaces: {e}")

        return "Submission is valid."