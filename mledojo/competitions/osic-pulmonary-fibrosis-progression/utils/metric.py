from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class OsicPulmonaryFibrosisProgressionMetrics(CompetitionMetrics):
    """
    Metric for OSIC Pulmonary Fibrosis Progression competition using a modified Laplace Log Likelihood.
    
    The score for each prediction is computed as:
    
        sigma_clipped = max(Confidence, 70)
        Delta = min(|FVC_true - FVC_predicted|, 1000)
        score = - (sqrt(2) * Delta / sigma_clipped) - ln(sqrt(2) * sigma_clipped)
    
    The final score is the average of the score over all Patient_Weeks.
    
    Note: Although score values are negative, a higher score is considered better.
    """
    def __init__(self, value: str = "FVC", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        prepared_data = self._prepare_for_metric(y_pred, y_true)
        fvc_true = prepared_data["fvc_true"]
        fvc_pred = prepared_data["fvc_pred"]
        confidence = prepared_data["confidence"]
        
        return self._laplace_log_likelihood(fvc_true, fvc_pred, confidence)

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        
        # Check required columns in ground truth
        assert "Patient_Week" in ground_truth.columns, "Ground truth DataFrame must have a 'Patient_Week' column."
        assert "FVC" in ground_truth.columns, "Ground truth DataFrame must have a 'FVC' column."
        assert "Patient" in ground_truth.columns, "Ground truth DataFrame must have a 'Patient' column."
        
        # Check required columns in submission
        if "Patient_Week" not in submission.columns:
            raise InvalidSubmissionError("Submission DataFrame must have a 'Patient_Week' column.")
        if "FVC" not in submission.columns:
            raise InvalidSubmissionError("Submission DataFrame must have a 'FVC' column.")
        if "Confidence" not in submission.columns:
            raise InvalidSubmissionError("Submission DataFrame must have a 'Confidence' column.")
        
        # Check if all Patient_Week values in submission exist in ground truth
        for pw in submission["Patient_Week"]:
            if pw not in ground_truth["Patient_Week"].values:
                raise InvalidSubmissionError(
                    f"Patient_Week {pw} in submission does not exist in ground truth"
                )
        
        # Check data types
        if not pd.api.types.is_numeric_dtype(submission["FVC"]):
            raise InvalidSubmissionError("FVC column in submission must be numeric.")
        if not pd.api.types.is_numeric_dtype(submission["Confidence"]):
            raise InvalidSubmissionError("Confidence column in submission must be numeric.")
        
        return "Submission is valid."
    
    def _prepare_for_metric(self, submission: pd.DataFrame, answers: pd.DataFrame) -> dict:
        # fillna with 0 for the confidence column
        submission["Confidence"] = submission["Confidence"].fillna(0)

        # We should only take the 3 latest Patient_Week records for each patient
        answers = answers.dropna(subset=["FVC"])  # Drop the dummy data
        answers["Week_Number"] = answers["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))
        latest_weeks = answers.sort_values("Week_Number").groupby("Patient").tail(3)
        answers = latest_weeks.drop(columns=["Week_Number"])
        
        # Make submission match; we only grade the prediction for the 3 latest weeks
        submission = submission[submission["Patient_Week"].isin(answers["Patient_Week"])]

        submission = submission.sort_values(by="Patient_Week")
        answers = answers.sort_values(by="Patient_Week")

        fvc_true = answers.loc[answers["Patient_Week"].isin(submission["Patient_Week"]), "FVC"].values
        fvc_pred = submission.loc[
            submission["Patient_Week"].isin(answers["Patient_Week"]), "FVC"
        ].values
        confidence = submission.loc[
            submission["Patient_Week"].isin(answers["Patient_Week"]), "Confidence"
        ].values

        return {"fvc_true": fvc_true, "fvc_pred": fvc_pred, "confidence": confidence}
    
    def _laplace_log_likelihood(
        self, actual_fvc: np.ndarray, predicted_fvc: np.ndarray, confidence: np.ndarray, return_values=False
    ) -> float:
        """
        Calculates the modified Laplace Log Likelihood score for osic-pulmonary-fibrosis-progression
        """
        sd_clipped = np.maximum(confidence, 70)
        delta = np.minimum(np.abs(actual_fvc - predicted_fvc), 1000)
        metric = -np.sqrt(2) * delta / sd_clipped - np.log(np.sqrt(2) * sd_clipped)

        if return_values:
            return metric
        else:
            return np.mean(metric)