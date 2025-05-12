#!/usr/bin/env python3
import os
import shutil
import pandas as pd
import numpy as np
from math import floor
from tqdm import tqdm
from pathlib import Path

def prepare(raw: Path, public: Path, private: Path):
    """
    Prepare data for the-winton-stock-market-challenge competition.
    
    Args:
        raw: Path to raw data directory
        public: Path to public data directory
        private: Path to private data directory
    """
    os.system("unzip -o " + str(raw / "train.csv.zip") + " -d " + str(raw))
    os.system("unzip -o " + str(raw / "test_2.csv.zip") + " -d " + str(raw))
    os.system("unzip -o " + str(raw / "sample_submission_2.csv.zip") + " -d " + str(raw))
    # Create output directories if they don't exist
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Step 1: Read original CSV files
    # -------------------------
    train_csv_path = raw / 'train.csv'
    test2_csv_path = raw / 'test_2.csv'

    train_df = pd.read_csv(train_csv_path)
    test2_df = pd.read_csv(test2_csv_path)

    n_train_original = len(train_df)
    n_test_original = len(test2_df)

    # Compute split ratio
    total_count = n_train_original + n_test_original
    r = n_test_original / total_count

    # -------------------------
    # Step 2: Split train.csv into new training and new test sets
    # -------------------------
    # Use fixed random seed for reproducibility
    new_train_df = train_df.sample(frac=(1 - r), random_state=42)
    new_test_df = train_df.drop(new_train_df.index)

    # Save the new training and test sets
    new_train_csv_path = public / 'train.csv'
    new_test_csv_path = public / 'test.csv'
    new_train_df.to_csv(new_train_csv_path, index=False)

    # -------------------------
    # Step 3: Build private/test_answer.csv from new test set
    # -------------------------
    # The target columns in each window are:
    # - Intraday returns: columns "Ret_121" to "Ret_180" (60 values)
    # - Extra return columns: "Ret_PlusOne" and "Ret_PlusTwo"
    intraday_cols = [f"Ret_{i}" for i in range(121, 181)]
    plus_cols = ["Ret_PlusOne", "Ret_PlusTwo"]

    test_answer_rows = []  # Will hold dicts with keys "Id" and "Answer"
    # Process each window in new test set using tqdm for progress visualization
    for window_idx, (_, row) in enumerate(tqdm(new_test_df.iterrows(), total=len(new_test_df), desc="Processing test_answer rows")):
        row_id = int(row["Id"])
        # Get answers in the required order
        answers = list(row[intraday_cols].values) + list(row[plus_cols].values)
        weight_intraday = row["Weight_Intraday"]
        weight_daily = row["Weight_Daily"]
        # For each answer, construct an id "windowIndex_predictionIndex"
        for pred_idx, ans in enumerate(answers, start=1):
            answer_row = {
                "Id": f"{row_id}_{pred_idx}",
                "Predicted": ans
            }
            # Add weights based on prediction index
            if 1 <= pred_idx <= 60:  # Intraday returns
                answer_row["Weight_Intraday"] = weight_intraday
                answer_row["Weight_Daily"] = 0.0
            else:  # Daily returns (pred_idx 61 or 62)
                answer_row["Weight_Intraday"] = 0.0
                answer_row["Weight_Daily"] = weight_daily
            
            test_answer_rows.append(answer_row)

    test_answer_df = pd.DataFrame(test_answer_rows)
    test_answer_csv_path = private / 'test_answer.csv'
    test_answer_df.to_csv(test_answer_csv_path, index=False)

    # -------------------------
    # Step 4: Create public/sample_submission.csv
    # -------------------------
    # The sample submission must have exactly the same Id rows as test_answer.csv 
    # and two columns: "Id" and "Predicted". Dummy predictions (0.0) will be used.
    sample_submission_df = pd.DataFrame({
        "Id": test_answer_df["Id"],
        "Predicted": 0.0
    })
    sample_submission_csv_path = public / 'sample_submission.csv'
    sample_submission_df.to_csv(sample_submission_csv_path, index=False)


    drop_cols = intraday_cols + plus_cols + ["Weight_Intraday", "Weight_Daily"]
    new_test_df.drop(columns=drop_cols).to_csv(new_test_csv_path, index=False)

    # -------------------------
    # Step 6: Assertions and Checks
    # -------------------------
    # A. Count Assertions
    new_train_count = len(new_train_df)
    new_test_count = len(new_test_df)
    assert new_train_count + new_test_count == n_train_original, "New train+test counts do not equal original train count."

    expected_new_test_count = floor(n_train_original * (n_test_original / (n_train_original + n_test_original)))
    # Allow for a difference of 1 due to rounding
    assert abs(new_test_count - expected_new_test_count) <= 1, "New test count does not match the expected ratio."

    # Check test_answer rows count equals new_test_count * 62
    assert len(test_answer_df) == new_test_count * 62, "Test answer rows count mismatch."

    # Check sample_submission.csv rows equal test_answer.csv rows
    sample_submission_loaded = pd.read_csv(sample_submission_csv_path)
    assert len(sample_submission_loaded) == len(test_answer_df), "Sample submission row count mismatch."

    # B. Header and Field Assertions
    sample_submission_header = list(sample_submission_loaded.columns)
    assert sample_submission_header == ["Id", "Predicted"], "sample_submission.csv header is incorrect."

    # Check each Id format in sample_submission (and test_answer) follows "number_index" where index is 1..62
    for id_val in sample_submission_loaded["Id"]:
        parts = id_val.split('_')
        assert len(parts) == 2, f"Id format incorrect: {id_val}"
        window_number, pred_index = parts
        try:
            w = int(window_number)
            p = int(pred_index)
            assert 1 <= p <= 62, f"Prediction index out of range in Id: {id_val}"
        except ValueError:
            assert False, f"Non-integer in Id: {id_val}"

    # Check dummy predictions are floats
    assert sample_submission_loaded["Predicted"].dtype in [np.float64, np.float32], "Dummy predictions are not floats."

    # C. Data Integrity Assertions
    # Verify that the new test set rows are disjoint from new train set rows (by index)
    common_indices = set(new_train_df.index).intersection(set(new_test_df.index))
    assert len(common_indices) == 0, "Overlap detected between new training and new test sets."

    # Verify that order is preserved: the first 62 rows in test_answer.csv correspond to first window in new_test.csv
    first_window_answers = test_answer_df.head(62)["Id"].tolist()
    expected_first_window_ids = [f"1_{i}" for i in range(1, 63)]
    assert first_window_answers == expected_first_window_ids, "Ordering of test_answer.csv Ids is incorrect."

    # D. File Structure Assertions
    assert os.path.isfile(private / 'test_answer.csv'), "private/test_answer.csv is missing."
    assert os.path.isfile(public / 'train.csv'), "public/train.csv is missing."
    assert os.path.isfile(public / 'test.csv'), "public/test.csv is missing."
    assert os.path.isfile(public / 'sample_submission.csv'), "public/sample_submission.csv is missing."

    # Ensure the original input folder remains unchanged (basic check: train.csv exists in input_dir)
    assert os.path.isfile(train_csv_path), "Original train.csv is missing in the source folder."

    print("Data reorganization completed successfully with all assertions passing.")

    # Clean up
    shutil.rmtree(raw)