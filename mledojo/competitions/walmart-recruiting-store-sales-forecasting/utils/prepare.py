#!/usr/bin/env python3
import os
import shutil
import pandas as pd
from tqdm import tqdm
import numpy as np
from pathlib import Path

def prepare(raw: Path, public: Path, private: Path):
    """
    Prepare data for walmart-recruiting-store-sales-forecasting competition.
    
    Args:
        raw: Path to raw data directory
        public: Path to public data directory
        private: Path to private data directory
    """
    os.system("unzip -o " + str(raw / "train.csv.zip") + " -d " + str(raw))
    os.system("unzip -o " + str(raw / "test.csv.zip") + " -d " + str(raw))
    os.system("unzip -o " + str(raw / "sampleSubmission.csv.zip") + " -d " + str(raw))
    os.system("unzip -o " + str(raw / "features.csv.zip") + " -d " + str(raw))
    # Create destination directories
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    
    # Set random seed for reproducibility
    RANDOM_SEED = 42
    
    # Step 1: Read original train.csv and test.csv
    train_path = raw / 'train.csv'
    orig_test_path = raw / 'test.csv'
    
    df_train = pd.read_csv(train_path)
    df_orig_test = pd.read_csv(orig_test_path)
    
    # Step 2: Compute split ratio
    N_train_orig = df_train.shape[0]
    N_test_orig = df_orig_test.shape[0]
    ratio = N_test_orig / (N_train_orig + N_test_orig)
    
    # Compute number of rows for new_test from train.csv
    new_test_count = int(round(ratio * N_train_orig))
    new_train_count = N_train_orig - new_test_count
    
    # Step 3: Shuffle train.csv rows with fixed seed
    df_train_shuffled = df_train.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    # Step 4: Split df_train_shuffled into New_Train and New_Test
    df_new_train = df_train_shuffled.iloc[:new_train_count].copy()
    df_new_test = df_train_shuffled.iloc[new_train_count:].copy()
    
    # Assertions for overall row counts
    assert df_new_train.shape[0] + df_new_test.shape[0] == N_train_orig, "Total rows after split do not sum to original."
    assert df_new_test.shape[0] == new_test_count, "Number of rows in New_Test does not match the expected count."
    
    # Step 5: Save New_Train as public/train.csv
    public_train_path = public / 'train.csv'
    df_new_train.to_csv(public_train_path, index=False)
    
    # Step 6: Save New_Test as public/test.csv (features file with answers intact)
    public_test_path = public / 'test.csv'
    df_new_test.drop(columns=["Weekly_Sales"]).to_csv(public_test_path, index=False)
    
    # Step 7: Create private/test_answer.csv from New_Test
    ids = []
    weekly_sales = []
    for idx, row in tqdm(df_new_test.iterrows(), total=df_new_test.shape[0], desc="Generating test_answer Ids"):
        cur_id = f"{str(row['Store'])}_{str(row['Dept'])}_{row['Date']}"
        ids.append(cur_id)
        weekly_sales.append(row['Weekly_Sales'])
    
    df_test_answer = pd.DataFrame({
        "Id": ids,
        "Weekly_Sales": weekly_sales
    })
    
    private_test_answer_path = private / 'test_answer.csv'
    df_test_answer.to_csv(private_test_answer_path, index=False)
    
    # Assertion: number of rows in public/test.csv equals number in private/test_answer.csv
    assert df_new_test.shape[0] == df_test_answer.shape[0], "Row count mismatch between test.csv and test_answer.csv."
    
    # Verify Id formation for each row (sample check)
    for i in range(min(10, len(df_new_test))):  # Check first 10 rows for efficiency
        expected_id = f"{str(df_new_test.iloc[i]['Store'])}_{str(df_new_test.iloc[i]['Dept'])}_{df_new_test.iloc[i]['Date']}"
        assert df_test_answer.iloc[i]['Id'] == expected_id, f"Id mismatch at row {i}"
    
    # Step 8: Create new sample_submission.csv in public/
    df_sample_submission = pd.DataFrame({
        "Id": df_test_answer["Id"],
        "Weekly_Sales": [0.0] * df_test_answer.shape[0]
    })
    public_sample_sub_path = public / 'sample_submission.csv'
    df_sample_submission.to_csv(public_sample_sub_path, index=False)
    
    # Assertions for sample_submission.csv
    expected_columns = ["Id", "Weekly_Sales"]
    assert list(df_sample_submission.columns) == expected_columns, "sample_submission.csv columns are not as expected."
    assert df_sample_submission.shape[0] == df_test_answer.shape[0], "sample_submission.csv row count does not match test_answer.csv."
    
    # Step 9: Copy files that are not split (features.csv and stores.csv) to public/
    files_to_copy = ['features.csv', 'stores.csv']
    for file_name in tqdm(files_to_copy, desc="Copying non-split files"):
        src_file = raw / file_name
        dest_file = public / file_name
        shutil.copy(src_file, dest_file)
    
    # Final assertion: Check that expected un-split files exist in public/
    for file_name in files_to_copy:
        dest_file = public / file_name
        assert os.path.exists(dest_file), f"{file_name} not found in public/ folder."
    
    print("Data reorganization completed successfully.")

    # Clean up
    shutil.rmtree(raw)