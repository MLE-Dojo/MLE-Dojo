#!/usr/bin/env python3
"""
This script reorganizes the original data into a new folder structure
with public and private splits according to the specified plan.

It will:
  – Determine the size of the local "test" split from test.csv.
  – Shuffle and split train.csv (and corresponding parquet files) into local train and test sets.
  – Create new CSV files for train and test features, along with a test answer CSV (with three rows per image)
    and a dummy sample_submission.csv matching its format.
  – Copy extra unsplit files to public directory.
  – Assert at multiple steps that the reorganized data is consistent.
"""

import os
import glob
import shutil
import pandas as pd
from tqdm import tqdm
from pathlib import Path

RANDOM_STATE = 42

def prepare(raw: Path, public: Path, private: Path):
    """
    Prepare the competition data by reorganizing it into public and private directories.
    
    Args:
        raw: Path to the raw data directory
        public: Path to the public output directory
        private: Path to the private output directory
    """
    # Create directory structure
    train_public_dir = os.path.join(public, 'train')
    test_public_dir = os.path.join(public, 'test')
    
    for directory in [train_public_dir, test_public_dir, private]:
        os.makedirs(directory, exist_ok=True)
    
    # Determine holdout size from test.csv
    test_csv_path = os.path.join(raw, 'test.csv')
    df_test = pd.read_csv(test_csv_path)
    n_holdout = len(df_test)
    print(f"n_holdout (number of rows in test.csv): {n_holdout}")
    
    # Split train.csv into train and test sets
    train_csv_path = os.path.join(raw, 'train.csv')
    df_train = pd.read_csv(train_csv_path)
    
    # Shuffle the rows reproducibly
    df_train_shuffled = df_train.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    
    # Get unique image_ids and split them
    image_ids = df_train_shuffled['image_id'].unique()
    assert len(image_ids) >= n_holdout, f"Not enough images in train.csv to create a test split of {n_holdout} images."
    
    test_image_ids = set(image_ids[:n_holdout])
    train_image_ids = set(image_ids[n_holdout:])
    
    # Create train and test splits
    df_train_split = df_train[df_train['image_id'].isin(train_image_ids)]
    df_test_split = df_train[df_train['image_id'].isin(test_image_ids)]
    
    # Verify split integrity
    assert len(df_train_split) + len(df_test_split) == len(df_train), "Row count mismatch after splitting train.csv"
    
    # Save new train.csv
    train_csv_out = os.path.join(train_public_dir, 'train.csv')
    df_train_split.to_csv(train_csv_out, index=False)
    
    # Save new test.csv (features only)
    test_features = df_test_split[['image_id']].copy()
    test_csv_out = os.path.join(test_public_dir, 'test.csv')
    test_features.to_csv(test_csv_out, index=False)
    
    # Process parquet files
    parquet_pattern = os.path.join(raw, 'train_image_data_*.parquet')
    parquet_files = glob.glob(parquet_pattern)
    
    if parquet_files:
        for parquet_file in tqdm(parquet_files, desc="Processing parquet files"):
            df = pd.read_parquet(parquet_file)
            total_rows = len(df)
            
            df_train_parquet = df[df['image_id'].isin(train_image_ids)]
            df_test_parquet = df[df['image_id'].isin(test_image_ids)]
            
            # Verify parquet split integrity
            common_ids = set(df_train_parquet['image_id'].unique()).intersection(set(df_test_parquet['image_id'].unique()))
            assert len(common_ids) == 0, f"Some image_id appears in both train and test splits in {parquet_file}"
            assert len(df_train_parquet) + len(df_test_parquet) == total_rows, f"Row count mismatch in parquet file {parquet_file}"
            
            base_filename = os.path.basename(parquet_file)
            df_train_parquet.to_parquet(os.path.join(train_public_dir, base_filename), index=False)
            df_test_parquet.to_parquet(os.path.join(test_public_dir, base_filename), index=False)
    
    # Build test_answer.csv
    components = ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']
    answer_rows = []
    
    for idx, row in df_test_split.reset_index(drop=True).iterrows():
        for comp in components:
            row_id = f"Test_{idx}_{comp}"
            target = row[comp]
            answer_rows.append({'row_id': row_id, 'target': target})
    
    df_test_answer = pd.DataFrame(answer_rows)
    
    # Verify test_answer integrity
    assert len(df_test_answer) == 3 * len(df_test_split), "test_answer.csv row count is not 3 * (number of test images)"
    assert df_test_answer['row_id'].is_unique, "Duplicate row_id detected in test_answer.csv"
    
    for rid in df_test_answer['row_id']:
        assert rid.startswith("Test_") and any(comp in rid for comp in components), f"row_id {rid} does not follow the required format."
    
    # Save test_answer.csv
    test_answer_out = os.path.join(private, 'test_answer.csv')
    df_test_answer.to_csv(test_answer_out, index=False)
    
    # Build sample_submission.csv
    df_submission = df_test_answer.copy()
    df_submission['target'] = 0  # Dummy answer
    
    # Verify sample_submission integrity
    assert len(df_submission) == len(df_test_answer), "Mismatch in row count between sample_submission and test_answer.csv"
    assert list(df_submission.columns) == ['row_id', 'target'], "Column names in sample_submission.csv are not ['row_id', 'target']"
    assert df_submission['row_id'].tolist() == df_test_answer['row_id'].tolist(), "row_id ordering mismatch"
    
    sample_submission_out = os.path.join(public, 'sample_submission.csv')
    df_submission.to_csv(sample_submission_out, index=False)
    
    # Copy unsplit files
    exclude_files = {'train.csv', 'test.csv', 'sample_submission.csv'}
    exclude_parquet_prefixes = ['train_image_data_', 'test_image_data_']
    
    for file_name in os.listdir(raw):
        file_path = os.path.join(raw, file_name)
        if os.path.isfile(file_path):
            if file_name in exclude_files:
                continue
            if any(file_name.startswith(prefix) and file_name.endswith('.parquet') for prefix in exclude_parquet_prefixes):
                continue
            target_path = os.path.join(public, file_name)
            shutil.copy2(file_path, target_path)
    
    # Final global assertions
    full_ids = set(df_train['image_id'].unique())
    assert full_ids == train_image_ids.union(test_image_ids), "The splitting of image_ids is not complete."
    
    new_train = pd.read_csv(os.path.join(train_public_dir, 'train.csv'))
    new_test = pd.read_csv(os.path.join(test_public_dir, 'test.csv'))
    unique_new_ids = set(new_train['image_id'].unique()).union(set(new_test['image_id'].unique()))
    assert unique_new_ids == full_ids, "Mismatch in total image_ids after splitting."
    
    df_test_features = pd.read_csv(os.path.join(test_public_dir, 'test.csv'))
    df_test_ans = pd.read_csv(os.path.join(private, 'test_answer.csv'))
    df_sample_sub = pd.read_csv(os.path.join(public, 'sample_submission.csv'))
    
    assert len(df_test_features) == len(test_image_ids), "Mismatch in number of test images in test.csv."
    assert len(df_test_ans) == 3 * len(test_image_ids), "test_answer.csv does not have 3 rows per test image."
    assert df_test_ans['row_id'].tolist() == df_sample_sub['row_id'].tolist(), "row_id ordering mismatch in test_answer.csv and sample_submission.csv."
    
    print(f"Total train image_ids: {len(train_image_ids)}; test image_ids: {len(test_image_ids)}")
    print("Data preparation completed successfully.")

    # Clean up
    shutil.rmtree(raw)
    # Delete zip file of public/
    for file in os.listdir(public):
        if file.endswith('.zip'):
            os.remove(os.path.join(public, file))