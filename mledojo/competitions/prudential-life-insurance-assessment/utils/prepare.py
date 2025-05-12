#!/usr/bin/env python3
import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

def prepare(raw: Path, public: Path, private: Path):
    """
    Prepare data for the prudential-life-insurance-assessment competition.
    
    Args:
        raw: Path to raw data directory
        public: Path to public data directory
        private: Path to private data directory
    """
    # Set a fixed random seed for reproducibility
    np.random.seed(42)
    os.system("unzip -o " + str(raw / "train.csv.zip") + " -d " + str(raw))
    os.system("unzip -o " + str(raw / "test.csv.zip") + " -d " + str(raw))
    os.system("unzip -o " + str(raw / "sample_submission.csv.zip") + " -d " + str(raw))
    
    # Create output directories
    public.mkdir(exist_ok=True, parents=True)
    private.mkdir(exist_ok=True, parents=True)
    
    # 1. Read original data files
    train_path = raw / "train.csv"
    test_comp_path = raw / "test.csv"
    
    print("Reading train.csv and competition test.csv ...")
    train_df = pd.read_csv(train_path)
    test_comp_df = pd.read_csv(test_comp_path)
    
    # 2. Compute hold-out size based on competition ratio
    N_orig = len(train_df)
    N_competition_test = len(test_comp_df)
    test_holdout_count = round(N_orig * (N_competition_test / (N_orig + N_competition_test)))
    print(f"Computed test hold-out size: {test_holdout_count} rows (from {N_orig} training rows).")
    
    # 3. Stratified split of train.csv
    # Verify required columns exist
    if "Response" not in train_df.columns:
        raise ValueError("Column 'Response' not found in train.csv")
    if "Id" not in train_df.columns:
        raise ValueError("Column 'Id' not found in train.csv")
    
    # Initialize lists to hold indices for the new test set
    selected_test_indices = []
    
    # Process each group with progress visualization
    print("Performing stratified sampling on train.csv ...")
    groups = list(train_df.groupby("Response"))
    for label, group in tqdm(groups, desc="Sampling groups"):
        # Calculate the number of samples to draw from this group
        n_group = len(group)
        n_sample = int(round(n_group * (test_holdout_count / N_orig)))
        n_sample = min(n_sample, n_group)
        sampled_idx = group.sample(n=n_sample, random_state=42).index.tolist()
        selected_test_indices.extend(sampled_idx)
    
    # Adjust sample size if needed due to rounding
    current_test_count = len(selected_test_indices)
    if current_test_count != test_holdout_count:
        diff = test_holdout_count - current_test_count
        all_indices = set(train_df.index)
        remaining_indices = list(all_indices - set(selected_test_indices))
        if diff > 0:
            # Need to add additional indices randomly
            extra_indices = np.random.choice(remaining_indices, size=abs(diff), replace=False).tolist()
            selected_test_indices.extend(extra_indices)
        elif diff < 0:
            # Remove extra indices randomly
            selected_test_indices = np.random.choice(selected_test_indices, size=test_holdout_count, replace=False).tolist()
    
    # Ensure the new test set indices are sorted
    selected_test_indices.sort()
    
    # Create new test and train DataFrames
    new_test_df = train_df.loc[selected_test_indices].copy()
    new_train_df = train_df.drop(index=selected_test_indices).copy()
    
    # 4. Assertions on the split
    assert len(new_train_df) + len(new_test_df) == N_orig, "Total rows after split do not equal the original count."
    assert list(new_train_df.columns) == list(train_df.columns), "Columns in new_train_df do not match original train.csv"
    assert list(new_test_df.columns) == list(train_df.columns), "Columns in new_test_df do not match original train.csv"
    
    # Verify response distribution
    orig_response_counts = train_df["Response"].value_counts().sort_index()
    new_train_response_counts = new_train_df["Response"].value_counts().sort_index()
    new_test_response_counts = new_test_df["Response"].value_counts().sort_index()
    for response in orig_response_counts.index:
        total = new_train_response_counts.get(response, 0) + new_test_response_counts.get(response, 0)
        assert total == orig_response_counts[response], f"Mismatch in counts for Response {response}"
    
    # 5. Save the new split files
    # Save new training set (features + answers)
    new_train_df.to_csv(public / "train.csv", index=False)
    
    # Save new test set (features only) in public folder
    new_test_df.drop(columns=['Response']).to_csv(public / "test.csv", index=False)
    
    # Save test answers (Id and Response) in private folder
    test_answer_df = new_test_df[["Id", "Response"]].copy()
    test_answer_df.to_csv(private / "test_answer.csv", index=False)
    
    # Verify test files consistency
    public_test_ids = new_test_df["Id"].tolist()
    private_test_ids = test_answer_df["Id"].tolist()
    assert len(public_test_ids) == len(private_test_ids), "Row counts differ between public test.csv and private test_answer.csv"
    assert public_test_ids == private_test_ids, "Id order mismatch between public test.csv and private test_answer.csv"
    
    # 6. Build sample_submission.csv for the new test set
    response_dtype = new_test_df["Response"].dtype
    dummy_value = 1  # Default dummy answer
    
    # Ensure the dummy value has the same type as the ground truth
    try:
        dummy_value = response_dtype.type(dummy_value)
    except Exception:
        pass
    
    sample_submission_df = pd.DataFrame({
        "Id": new_test_df["Id"],
        "Response": [dummy_value] * len(new_test_df)
    })
    
    # Verify submission file format
    assert list(sample_submission_df.columns) == ["Id", "Response"], "sample_submission.csv columns do not match ['Id', 'Response']"
    assert len(sample_submission_df) == len(test_answer_df), "sample_submission.csv row count does not match test_answer.csv"
    assert sample_submission_df["Response"].dtype == new_test_df["Response"].dtype, "Dummy answer data type does not match ground truth data type."
    
    # Save sample_submission.csv in public folder
    sample_submission_df.to_csv(public / "sample_submission.csv", index=False)
        
    
    print("Data preparation complete.")

    # Clean up
    shutil.rmtree(raw)
