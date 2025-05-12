#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def prepare(raw: Path, public: Path, private: Path):
    """
    Prepare data for the sf-crime competition.
    
    Args:
        raw: Path to raw data directory
        public: Path to public data directory
        private: Path to private data directory
    """
    os.system("unzip -o " + str(raw / "train.csv.zip") + " -d " + str(raw))
    os.system("unzip -o " + str(raw / "sampleSubmission.csv.zip") + " -d " + str(raw))
    os.system("unzip -o " + str(raw / "test.csv.zip") + " -d " + str(raw))

    # Create output directories
    public.mkdir(exist_ok=True, parents=True)
    private.mkdir(exist_ok=True, parents=True)
    
    # Define file paths
    train_csv_path = raw / "train.csv"
    sample_submission_orig_path = raw / "sampleSubmission.csv"
    
    # Output paths
    train_out_path = public / "train.csv"
    test_features_out_path = public / "test.csv"
    test_answer_out_path = private / "test_answer.csv"
    sample_submission_out_path = public / "sample_submission.csv"
    
    # 1. Process train.csv: split into train and test sets
    print("Reading train.csv ...")
    df_train = pd.read_csv(train_csv_path)
    
    # Split train data using sklearn
    df_local_train, df_local_test = train_test_split(df_train, test_size=0.2, random_state=42)
    
    # Assertions for splitting
    total_original = len(df_train)
    total_split = len(df_local_train) + len(df_local_test)
    assert total_original == total_split, "Split row count does not match original!"
    
    # Save local training set
    print("Writing local training set to:", train_out_path)
    df_local_train.to_csv(train_out_path, index=False)
    
    # 2. Create test_answer.csv from local test set
    # Read original sampleSubmission.csv to get candidate class names
    print("Reading original sampleSubmission.csv ...")
    df_sample_submission_orig = pd.read_csv(sample_submission_orig_path)
    
    if df_sample_submission_orig.columns[0] != "Id":
        raise ValueError("The first column of sampleSubmission.csv must be 'Id'.")
    candidate_classes = list(df_sample_submission_orig.columns[1:])
    
    # Generate an 'Id' column for test set
    df_local_test = df_local_test.reset_index(drop=True)
    df_local_test.insert(0, "Id", df_local_test.index)
    
    # Create test features file
    df_test_features = df_local_test.copy()
    
    # Create test_answer dataframe with one-hot encoding
    test_answer_rows = []
    print("Constructing test answer one-hot encoding ...")
    for idx, row in tqdm(df_local_test.iterrows(), total=len(df_local_test)):
        true_category = row["Category"]
        # Initialize one-hot vector
        one_hot = {cls: 0 for cls in candidate_classes}
        if true_category not in candidate_classes:
            raise ValueError(f"Found unexpected category '{true_category}' in row Id {row['Id']}.")
        one_hot[true_category] = 1
        # Verify that exactly one column is set to 1
        if sum(one_hot.values()) != 1:
            raise ValueError(f"One-hot encoding error in row Id {row['Id']}.")
        row_dict = {"Id": row["Id"]}
        row_dict.update(one_hot)
        test_answer_rows.append(row_dict)
    
    df_test_answer = pd.DataFrame(test_answer_rows, columns=["Id"] + candidate_classes)
    
    # Assertion: header must match expected format
    expected_header = ["Id"] + candidate_classes
    assert list(df_test_answer.columns) == expected_header, "Header of test_answer.csv does not match expected format!"
    assert len(df_test_answer) == len(df_local_test), "Row count in test_answer.csv does not match local test set!"
    
    # Save test_answer.csv
    print("Writing test answer file to:", test_answer_out_path)
    df_test_answer.to_csv(test_answer_out_path, index=False)
    
    # 3. Create sample_submission.csv with dummy predictions
    df_sample_submission = df_test_answer.copy()
    for cls in candidate_classes:
        df_sample_submission[cls] = 0.0  # Ensure float type
    
    # Assertions for sample_submission
    assert list(df_sample_submission.columns) == expected_header, "Header of sample_submission.csv does not match expected format!"
    assert len(df_sample_submission) == len(df_test_answer), "Row count in sample_submission.csv does not match test_answer.csv!"
    
    print("Writing sample submission file to:", sample_submission_out_path)
    df_sample_submission.to_csv(sample_submission_out_path, index=False)
    
    # 4. Final assertions
    total_output_records = len(df_local_train) + len(df_local_test)
    assert total_output_records == total_original, "Total records in split files do not equal original count!"
    
    # Verify 'Id' columns match across files
    ids_test_features = df_test_features["Id"].tolist()
    ids_test_answer = df_test_answer["Id"].tolist()
    ids_sample_submission = df_sample_submission["Id"].tolist()
    assert ids_test_features == ids_test_answer == ids_sample_submission, "Mismatch in 'Id' columns among test files!"
    
    # Save test features without answer columns
    print("Writing local test features to:", test_features_out_path)
    df_test_features.drop(columns=["Category", "Descript", "Resolution"]).to_csv(test_features_out_path, index=False)
    
    print("Data reorganization complete. All assertions passed.")

    # Clean up
    shutil.rmtree(raw)