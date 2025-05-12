from typing import Any
import pandas as pd
import numpy as np
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError
import ast # Import ast for literal_eval

class NbmeScoreClinicalPatientNotesMetrics(CompetitionMetrics):
    """Metric class for NBME Score Clinical Patient Notes competition using micro-averaged F1 score."""
    def __init__(self, value: str = "location", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure the id columns (first column) are string type and sort by them
        y_true[y_true.columns[0]] = y_true[y_true.columns[0]].astype(str)
        y_pred[y_pred.columns[0]] = y_pred[y_pred.columns[0]].astype(str)
        y_true = y_true.sort_values(by=y_true.columns[0]).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=y_pred.columns[0]).reset_index(drop=True)
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        # Helper function to parse spans string to a set of character indices
        def parse_spans(span_str):
            indices = set()
            # Handle NaN, empty string, or literal "[]"
            if pd.isna(span_str) or str(span_str).strip() == "" or str(span_str).strip() == "[]":
                return indices

            processed_str = str(span_str).strip()
            span_segments = []

            # Check if it looks like a string representation of a list
            if processed_str.startswith('[') and processed_str.endswith(']'):
                try:
                    # Safely evaluate the string as a Python literal (list)
                    potential_list = ast.literal_eval(processed_str)
                    if isinstance(potential_list, list):
                        # If it's a list, use its elements as the segments
                        span_segments = potential_list
                    else:
                        # If literal_eval resulted in something else, treat the original string as one segment
                        span_segments = [processed_str]
                except (ValueError, SyntaxError):
                    # If parsing fails, treat the original string as one segment
                    span_segments = [processed_str]
            else:
                # If not list-like, treat the whole string as one segment
                span_segments = [processed_str]

            # Process each segment (either from the list or the single original string)
            for segment in span_segments:
                segment = str(segment).strip() # Ensure segment is string
                # Each span within a segment is separated by ;
                spans = segment.split(';')
                for span in spans:
                    span = span.strip()
                    if span == "":
                        continue
                    parts = span.split()
                    # Ensure span consists of exactly two parts
                    if len(parts) != 2:
                        # Optionally, log or warn about invalid span format
                        # print(f"Warning: Skipping invalid span format '{span}'")
                        continue
                    try:
                        start, end = int(parts[0]), int(parts[1])
                        # Ensure start is less than or equal to end
                        if start > end:
                            # Optionally, log or warn about invalid span range
                            # print(f"Warning: Skipping invalid span range '{span}' where start > end")
                            continue
                    except ValueError:
                        # Optionally, log or warn about non-integer values
                        # print(f"Warning: Skipping span '{span}' with non-integer values")
                        continue  # skip invalid numbers
                    # In Python, span [i:j] covers indices from i to j-1
                    indices.update(range(start, end))
            return indices
        
        # Iterate row-wise to aggregate true positives, false positives, and false negatives
        for idx in range(len(y_true)):
            true_spans = parse_spans(y_true.loc[idx, self.value])
            pred_spans = parse_spans(y_pred.loc[idx, self.value])
            
            tp = len(true_spans & pred_spans)
            fp = len(pred_spans - true_spans)
            fn = len(true_spans - pred_spans)
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
        
        denominator = (2 * total_tp) + total_fp + total_fn
        # If there are no ground truth spans and no predicted spans, define F1 as 1.0
        if denominator == 0:
            return 1.0
        score = (2 * total_tp) / denominator
        return score

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(
                f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows."
            )

        # Convert the id columns to string type in both submission and ground truth
        submission[submission.columns[0]] = submission[submission.columns[0]].astype(str)
        ground_truth[ground_truth.columns[0]] = ground_truth[ground_truth.columns[0]].astype(str)
        # Sort by the id column
        submission = submission.sort_values(by=submission.columns[0]).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=ground_truth.columns[0]).reset_index(drop=True)

        # Check if the id columns are identical
        if (submission[submission.columns[0]].values != ground_truth[ground_truth.columns[0]].values).any():
            raise InvalidSubmissionError(
                "The id column values do not match between submission and ground truth. Please ensure the first column values are identical and in the same order."
            )

        submission_cols = set(submission.columns)
        ground_truth_cols = set(ground_truth.columns)

        missing_cols = ground_truth_cols - submission_cols
        extra_cols = submission_cols - ground_truth_cols

        if missing_cols:
            raise InvalidSubmissionError(
                f"Missing required columns in submission: {', '.join(missing_cols)}."
            )
        if extra_cols:
            raise InvalidSubmissionError(
                f"Extra unexpected columns found in submission: {', '.join(extra_cols)}."
            )

        return "Submission is valid."