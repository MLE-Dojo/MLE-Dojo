from typing import Any
import pandas as pd
from mledojo.metrics.base import CompetitionMetrics, InvalidSubmissionError

class Ai4CodeMetrics(CompetitionMetrics):
    """Metric class for AI4Code competition using Kendall tau correlation based on number of adjacent swaps."""
    def __init__(self, value: str = "cell_order", higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.value = value

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        # Ensure the id column (first column) is string
        y_true[y_true.columns[0]] = y_true[y_true.columns[0]].astype(str)
        y_pred[y_pred.columns[0]] = y_pred[y_pred.columns[0]].astype(str)

        # Sort both dataframes by id (the first column)
        y_true = y_true.sort_values(by=y_true.columns[0]).reset_index(drop=True)
        y_pred = y_pred.sort_values(by=y_pred.columns[0]).reset_index(drop=True)

        total_inversions = 0
        total_denominator = 0

        # Iterate over each row (each notebook)
        for idx in range(len(y_true)):
            # # Retrieve the notebook id (first column)
            # notebook_id_true = y_true.iloc[idx, 0]
            # notebook_id_pred = y_pred.iloc[idx, 0]
            # if notebook_id_true != notebook_id_pred:
            #     raise InvalidSubmissionError(f"Notebook id mismatch at row {idx}: {notebook_id_true} != {notebook_id_pred}")

            # Get ground truth and predicted cell orders as list of cell ids
            true_order_str = y_true.iloc[idx][self.value]
            pred_order_str = y_pred.iloc[idx][self.value]

            true_order = true_order_str.strip().split()
            pred_order = pred_order_str.strip().split()

            # if set(true_order) != set(pred_order):
            #     raise InvalidSubmissionError(f"Cell ids mismatch for notebook id {notebook_id_true}. Ensure predicted and true cell orders contain the same cell ids.")

            # Create mapping from cell id to its position in the ground truth
            pos = {cell_id: i for i, cell_id in enumerate(true_order)}
            # Convert the predicted order into a list of ranks based on ground truth order
            rank_list = [pos[cell_id] for cell_id in pred_order]

            # Count inversions (i.e., number of adjacent swaps needed)
            inversions = self._count_inversions(rank_list)
            total_inversions += inversions

            n = len(true_order)
            total_denominator += n * (n - 1)

        # If there were no notebooks or denominators are zero, return perfect score
        if total_denominator == 0:
            return 1.0

        score = 1 - 4 * (total_inversions / total_denominator)
        return score

    def _count_inversions(self, arr: list) -> int:
        # Uses a modified merge sort algorithm to count inversions
        def merge_sort_count(a):
            if len(a) <= 1:
                return a, 0
            mid = len(a) // 2
            left, inv_left = merge_sort_count(a[:mid])
            right, inv_right = merge_sort_count(a[mid:])
            merged, inv_split = merge(left, right)
            return merged, inv_left + inv_right + inv_split

        def merge(left, right):
            i = 0
            j = 0
            merged = []
            inv_count = 0
            while i < len(left) and j < len(right):
                if left[i] <= right[j]:
                    merged.append(left[i])
                    i += 1
                else:
                    merged.append(right[j])
                    inv_count += len(left) - i
                    j += 1
            merged.extend(left[i:])
            merged.extend(right[j:])
            return merged, inv_count

        _, count = merge_sort_count(arr)
        return count

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise InvalidSubmissionError("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise InvalidSubmissionError("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise InvalidSubmissionError(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Convert the first column (id column) to string type
        submission[submission.columns[0]] = submission[submission.columns[0]].astype(str)
        ground_truth[ground_truth.columns[0]] = ground_truth[ground_truth.columns[0]].astype(str)
        # Sort both submission and ground truth by the id column
        submission = submission.sort_values(by=submission.columns[0]).reset_index(drop=True)
        ground_truth = ground_truth.sort_values(by=ground_truth.columns[0]).reset_index(drop=True)

        # Verify that the 'id' columns match exactly
        if not (submission[submission.columns[0]].values == ground_truth[ground_truth.columns[0]].values).all():
            raise InvalidSubmissionError("The id values in submission do not match those in ground truth. Please ensure the first column values are identical and in the same order.")

        required_cols = {submission.columns[0], self.value}
        sub_cols = set(submission.columns)
        true_cols = set(ground_truth.columns)

        missing_cols = required_cols - sub_cols
        extra_cols = sub_cols - required_cols

        if missing_cols:
            raise InvalidSubmissionError(f"Missing required columns in submission: {', '.join(missing_cols)}.")
        if extra_cols:
            raise InvalidSubmissionError(f"Extra unexpected columns found in submission: {', '.join(extra_cols)}.")

        # Validate cell ids match for each notebook
        for idx in range(len(submission)):
            notebook_id_true = ground_truth.iloc[idx][ground_truth.columns[0]]
            true_order_str = ground_truth.iloc[idx][self.value]
            pred_order_str = submission.iloc[idx][self.value]
            try:
                true_order = true_order_str.strip().split()
                pred_order = pred_order_str.strip().split()
            except:
                raise InvalidSubmissionError(f"The format of the cell order is incorrect for notebook id {notebook_id_true}.")

            if set(true_order) != set(pred_order):
                raise InvalidSubmissionError(f"Cell ids mismatch for notebook id {notebook_id_true}. Ensure predicted and true cell orders contain the same cell ids.")

        return "Submission is valid."