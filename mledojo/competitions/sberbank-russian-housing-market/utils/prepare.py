#!/usr/bin/env python3
import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def prepare(raw: Path, public: Path, private: Path):
    """
    Prepare data for the sberbank-russian-housing-market competition.
    
    Args:
        raw: Path to raw data directory
        public: Path to public data directory
        private: Path to private data directory
    """
    os.system("unzip -o " + str(raw / "train.csv.zip") + " -d " + str(raw))
    os.system("unzip -o " + str(raw / "test.csv.zip") + " -d " + str(raw))
    os.system("unzip -o " + str(raw / "macro.csv.zip") + " -d " + str(raw))
    os.system("unzip -o " + str(raw / "sample_submission.csv.zip") + " -d " + str(raw))


    # Create the organized folder structure
    public.mkdir(exist_ok=True, parents=True)
    private.mkdir(exist_ok=True, parents=True)

    # Process train.csv: Read, sort, and split into new training and test sets
    train_csv_path = raw / 'train.csv'
    df_train = pd.read_csv(train_csv_path)

    # Sort rows by the "timestamp" column
    df_train.sort_values(by='timestamp', inplace=True)

    # Determine split sizes (46 months for training, 11 months for internal test)
    total_rows = len(df_train)
    n_test = round(total_rows * (11/57))
    n_train = total_rows - n_test

    # Split the data
    df_new_train = df_train.iloc[:n_train].copy()
    df_new_test = df_train.iloc[n_train:].copy()

    # Assertions
    assert len(df_new_train) + len(df_new_test) == total_rows, "Split row counts do not match total rows."
    if not df_new_train.empty and not df_new_test.empty:
        assert df_new_test['timestamp'].min() >= df_new_train['timestamp'].max(), "Test set timestamps are not later than train set."

    # Save new training and test CSV files
    df_new_train.to_csv(public / 'train.csv', index=False)
    df_new_test.drop(columns=['price_doc']).to_csv(public / 'test.csv', index=False)

    # Build the organized internal test data from train.csv split
    # Create private/test_answer.csv containing "id" and "price_doc" for the test set
    df_test_answer = df_new_test[['id', 'price_doc']].copy()
    df_test_answer.to_csv(private / 'test_answer.csv', index=False)

    # Create public/sample_submission.csv based on test_answer.csv with dummy numerical values
    df_sample_submission = df_test_answer.copy()
    df_sample_submission['price_doc'] = 0.0
    df_sample_submission.to_csv(public / 'sample_submission.csv', index=False)

    # Copy unsplit supplementary files
    files_to_copy = ['macro.csv', 'data_dictionary.txt']
    for file_name in tqdm(files_to_copy, desc="Copying unsplit files"):
        src_path = raw / file_name
        if src_path.exists():
            shutil.copy(src_path, public / file_name)

    print("Data reorganization completed successfully.")

    # Clean up
    shutil.rmtree(raw)