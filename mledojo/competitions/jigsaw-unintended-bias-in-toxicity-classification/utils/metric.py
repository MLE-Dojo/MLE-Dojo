from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError
from sklearn.metrics import roc_auc_score

def safe_auc(y_true, y_score):
    # y_true should be binary (0/1) values
    if (np.unique(y_true).size == 1):
        return np.nan
    return roc_auc_score(y_true, y_score)

def generalized_power_mean(metric_list, p=-5):
    # Remove nan values
    valid_metrics = [m for m in metric_list if not np.isnan(m)]
    if len(valid_metrics) == 0:
        return np.nan
    valid_metrics = np.array(valid_metrics)
    return np.power(np.mean(np.power(valid_metrics, p)), 1/p)

class JigsawUnintendedBiasInToxicityClassificationMetrics(CompetitionMetrics):
    """Metric class for Jigsaw Unintended Bias in Toxicity Classification competition.
    
    The final score is computed by combining the overall AUC and a generalized mean
    (with power p=-5) of three bias AUCs (Subgroup AUC, BPSN AUC, and BNSP AUC), each weighted equally.
    See: https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/overview/evaluation
    """
    def __init__(self, value: str = "prediction", higher_is_better: bool = True, toxicity_col: str = "toxicity", p: int = -5, w: float = 0.25):
        super().__init__(higher_is_better)
        self.value = value  # prediction column in submission
        self.toxicity_col = toxicity_col # target column in ground truth
        self.p = p # Power for generalized mean
        self.w = w # Weight for each component
        # Default identity columns used for bias evaluation, matching the Kaggle setup
        self.identity_columns = [
            "male", "female", "transgender", "other_gender", "heterosexual",
            "homosexual_gay_or_lesbian", "bisexual", "other_sexual_orientation",
            "christian", "jewish", "muslim", "hindu", "buddhist", "atheist",
            "other_religion", "black", "white", "asian", "latino", "other_race_or_ethnicity",
            "physical_disability", "intellectual_or_learning_disability",
            "psychiatric_or_mental_illness", "other_disability"
        ]


    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        id_col = y_true.columns[0] # Assume first column is id
        
        # Select only necessary columns from ground truth
        required_true_cols = [id_col, self.toxicity_col] + [col for col in self.identity_columns if col in y_true.columns]
        y_true_subset = y_true[required_true_cols].copy()
        
        # Convert id columns to string and sort by id
        y_true_subset[id_col] = y_true_subset[id_col].astype(str)
        y_pred[id_col] = y_pred[id_col].astype(str)
        y_true_subset = y_true_subset.sort_values(by=id_col).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=id_col).reset_index(drop=True)
        
        # Merge on id column to ensure alignment
        # Use suffixes to distinguish columns from y_true and y_pred
        merged = pd.merge(y_true_subset, y_pred[[id_col, self.value]], on=id_col)
        
        # Define column names after merge
        target_col_true = self.toxicity_col
        pred_col = self.value # Now prediction column name is just self.value
        
        # Check if all required identity columns are present in the ground truth
        present_identity_columns = [col for col in self.identity_columns if col in merged.columns]
        if not present_identity_columns:
             # Or handle appropriately, maybe return 0 or raise error if bias calculation is impossible
             print("Warning: No identity columns found in ground truth. Bias AUCs cannot be calculated.")


        # Binarize the true toxicity label using >= 0.5 threshold for AUC calculation
        true_labels_binary = (merged[target_col_true] >= 0.5).astype(int)
        
        # Calculate overall AUC using the binarized true labels and raw prediction scores
        overall_auc = safe_auc(true_labels_binary, merged[pred_col])
        
        subgroup_auc_list = []
        bpsn_auc_list = []
        bnsp_auc_list = []
        
        # Loop through each identity column present in the data and compute bias AUCs
        for col in present_identity_columns:
            identity_col = col # Name in merged dataframe

            # Binarize the identity column for filtering using >= 0.5 threshold
            identity_present = (merged[identity_col] >= 0.5)
            identity_absent = ~identity_present
            target_toxic = true_labels_binary == 1
            target_nontoxic = ~target_toxic

            # Subgroup AUC: examples that mention the identity
            subgroup_mask = identity_present
            if subgroup_mask.any():
                subgroup_auc = safe_auc(true_labels_binary[subgroup_mask], merged.loc[subgroup_mask, pred_col])
                subgroup_auc_list.append(subgroup_auc)
            else:
                 subgroup_auc_list.append(np.nan) # Add NaN if subgroup is empty
            
            # BPSN AUC: Background Positive (toxic examples that do NOT mention the identity) & 
            #           Subgroup Negative (non-toxic examples that DO mention the identity).
            bpsn_mask = (target_toxic & identity_absent) | (target_nontoxic & identity_present)
            if bpsn_mask.any():
                bpsn_auc = safe_auc(true_labels_binary[bpsn_mask], merged.loc[bpsn_mask, pred_col])
                bpsn_auc_list.append(bpsn_auc)
            else:
                bpsn_auc_list.append(np.nan) # Add NaN if subgroup is empty

            # BNSP AUC: Background Negative (non-toxic examples that do NOT mention the identity) &
            #           Subgroup Positive (toxic examples that DO mention the identity).
            bnsp_mask = (target_nontoxic & identity_absent) | (target_toxic & identity_present)
            if bnsp_mask.any():
                bnsp_auc = safe_auc(true_labels_binary[bnsp_mask], merged.loc[bnsp_mask, pred_col])
                bnsp_auc_list.append(bnsp_auc)
            else:
                bnsp_auc_list.append(np.nan) # Add NaN if subgroup is empty
        
        # Compute generalized means for each set of bias AUCs using power p
        gen_mean_subgroup_auc = generalized_power_mean(subgroup_auc_list, p=self.p)
        gen_mean_bpsn_auc = generalized_power_mean(bpsn_auc_list, p=self.p)
        gen_mean_bnsp_auc = generalized_power_mean(bnsp_auc_list, p=self.p)
        
        # Handle cases where a generalized mean calculation resulted in NaN (e.g., all inputs were NaN)
        # Replace NaN generalized means with 0 for the final score calculation, or propagate NaN if preferred.
        # Here we replace with 0, assuming a missing bias component contributes 0 to the score.
        bias_components = [gen_mean_subgroup_auc, gen_mean_bpsn_auc, gen_mean_bnsp_auc]
        valid_bias_components = [comp for comp in bias_components if not np.isnan(comp)]

        # Final metric: weighted sum using weight w
        # Ensure overall_auc is not NaN before including it. If it is, the score is likely 0 or undefined.
        if np.isnan(overall_auc):
             overall_auc = 0.0 # Treat NaN overall AUC as 0 contribution

        final_score = self.w * overall_auc + sum(self.w * comp for comp in valid_bias_components)
        
        # Handle case where the final score is NaN (e.g., overall AUC was NaN and no valid bias components)
        if np.isnan(final_score):
             return 0.0 # Default to 0.0 if the final score calculation results in NaN

        return final_score

    def validate_submission(self, submission: pd.DataFrame, ground_truth: pd.DataFrame) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError(
                "Submission must be a pandas DataFrame."
            )
        if not isinstance(ground_truth, pd.DataFrame):
            # Although not strictly needed for validation, it's good practice
            raise InvalidSubmissionError(
                "Ground truth must be a pandas DataFrame."
            )

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(
                f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)})."
            )
        
        # Assume first column is the ID column for both
        if submission.shape[1] == 0:
             raise InvalidSubmissionError("Submission DataFrame is empty.")
        if ground_truth.shape[1] == 0:
             raise InvalidSubmissionError("Ground Truth DataFrame is empty.")
             
        id_col_sub = submission.columns[0]
        id_col_gt = ground_truth.columns[0]

        # Check for required columns in submission: id and prediction column
        required_columns = {id_col_sub, self.value}
        submission_columns = set(submission.columns)
        
        # Ensure the submission ID column has the same name as the ground truth ID column for consistency check during validation
        if id_col_sub != id_col_gt:
             raise InvalidSubmissionError(
                 f"Submission ID column name ('{id_col_sub}') must match Ground Truth ID column name ('{id_col_gt}')."
             )
        
        missing_cols = required_columns - submission_columns
        # Allow only the required columns (ID and prediction value)
        extra_cols = submission_columns - required_columns

        if missing_cols:
             # This error message implicitly covers the case where id_col_sub is missing
             raise InvalidSubmissionError(
                 f"Missing required columns in submission: {', '.join(missing_cols)}. Required columns are '{id_col_gt}' and '{self.value}'."
             )
        if extra_cols:
            raise InvalidSubmissionError(
                f"Extra unexpected columns found in submission: {', '.join(extra_cols)}. Required columns are '{id_col_gt}' and '{self.value}'."
            )
            
        # Check prediction column data type (should contain raw scores)
        if not pd.api.types.is_numeric_dtype(submission[self.value]):
             raise InvalidSubmissionError(
                 f"Prediction column '{self.value}' must contain numeric values."
             )

        # Convert id column to string type for comparison
        try:
            submission_ids = submission[id_col_sub].astype(str)
            ground_truth_ids = ground_truth[id_col_gt].astype(str)
        except Exception as e:
             raise InvalidSubmissionError(f"Could not convert ID columns to string: {e}")

        # Check if id sets are identical
        if set(submission_ids) != set(ground_truth_ids):
             raise InvalidSubmissionError(
                 f"The set of IDs in the '{id_col_sub}' column does not match the set of IDs in the ground truth."
             )

        # Check if id columns are identical *after sorting* to ensure row alignment for evaluation
        submission_ids_sorted = submission_ids.sort_values().reset_index(drop=True)
        ground_truth_ids_sorted = ground_truth_ids.sort_values().reset_index(drop=True)

        if not submission_ids_sorted.equals(ground_truth_ids_sorted):
            # This case should ideally be caught by the set check, but explicit check after sorting is safer
             raise InvalidSubmissionError(
                 f"The '{id_col_sub}' column values do not match between submission and ground truth after sorting. Please ensure the ID columns contain the exact same values."
             )


        return "Submission is valid."