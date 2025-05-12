from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError
from sklearn.metrics import recall_score

class BengaliaiCv19Metrics(CompetitionMetrics):
    """
    Metric class for the bengaliai-cv19 competition.
    The score is computed as a weighted average of the three macro-averaged recall scores for the 
    grapheme_root, consonant_diacritic, and vowel_diacritic components, with the grapheme_root having double weight.
    """
    def __init__(self, value: str = "target", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Convert row_id to string type
        y_true["row_id"] = y_true["row_id"].astype(str)
        y_pred["row_id"] = y_pred["row_id"].astype(str)
        # Sort both dataframes by row_id
        y_true = y_true.sort_values(by="row_id").reset_index(drop=True)
        y_pred = y_pred.sort_values(by="row_id").reset_index(drop=True)

        # Extract the component from row_id (expected format: Test_{num}_{component})
        # Extract the component from row_id (format: Test_{num}_{component})
        # For example, from "Test_0_grapheme_root" extract "grapheme_root"
        y_true["component"] = y_true["row_id"].apply(lambda x: x.split("_", 2)[2] if len(x.split("_")) > 2 else x.split("_")[-1])
        y_pred["component"] = y_pred["row_id"].apply(lambda x: x.split("_", 2)[2] if len(x.split("_")) > 2 else x.split("_")[-1])

        # Define the components and corresponding weights: grapheme_root gets weight 2, others get weight 1
        components = ["grapheme_root", "consonant_diacritic", "vowel_diacritic"]
        weights = [2, 1, 1]
        scores = []

        for comp in components:
            true_subset = y_true[y_true["component"] == comp][self.value].values
            pred_subset = y_pred[y_pred["component"] == comp][self.value].values

            # Calculate macro-averaged recall for the component
            comp_recall = recall_score(true_subset, pred_subset, average="macro")
            scores.append(comp_recall)

        final_score = np.average(scores, weights=weights)
        return final_score

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Convert row_id to string type for both submission and ground truth
        submission["row_id"] = submission["row_id"].astype(str)
        ground_truth["row_id"] = ground_truth["row_id"].astype(str)
        
        # Sort both by row_id        submission = submission.sort_values(by="row_id").reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by="row_id").reset_index(drop=True)
        submission = submission.sort_values(by="row_id").reset_index(drop=True)

        # Check if the row_id values match exactly
        if not (submission["row_id"].values == ground_truth["row_id"].values).all():
            raise InvalidSubmissionError("Row IDs in submission do not match those in ground truth. Please ensure the row_id values are identical and in the same order.")

        required_columns = {"row_id", self.value}
        submission_cols = set(submission.columns)
        missing_cols = required_columns - submission_cols
        extra_cols = submission_cols - required_columns

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        return "Submission is valid."