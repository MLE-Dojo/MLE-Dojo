#!/usr/bin/env python3
import os
import math
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import shutil

def prepare(raw: Path, public: Path, private: Path):
    """
    Prepare data for the poker-rule-induction competition.
    
    Args:
        raw: Path to raw data directory
        public: Path to public data directory
        private: Path to private data directory
    """
    # Create organized_data folder structure if it doesn't exist
    public.mkdir(exist_ok=True, parents=True)
    private.mkdir(exist_ok=True, parents=True)
    
    os.system("unzip -o " + str(raw / "train.csv.zip") + " -d " + str(raw))
    os.system("unzip -o " + str(raw / "test.csv.zip") + " -d " + str(raw))
    os.system("unzip -o " + str(raw / "sampleSubmission.csv.zip") + " -d " + str(raw))
    # Read original train.csv
    df = pd.read_csv(raw / "train.csv")
    
    # Get original header (column order), and number of rows in train.csv
    original_columns = list(df.columns)
    total_rows = df.shape[0]
    
    # Define split ratios (Competition ratios: local_training_ratio ≈ 0.0244)
    local_training_ratio = 0.0244
    n_train = int(round(total_rows * local_training_ratio))
    
    # Shuffle the dataframe with a fixed random seed for reproducibility
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split the data into a small training set and a large test set
    df_local_train = df_shuffled.iloc[:n_train].copy()
    df_local_test = df_shuffled.iloc[n_train:].copy()
    
    # Write the local training set to public/train.csv
    df_local_train.to_csv(public / "train.csv", index=False)
    
    # Write the local test set (features only) to public/test.csv
    df_local_test.drop(columns=['hand']).to_csv(public / "test.csv", index=False)
    
    # Create test_answer.csv with "id" and "hand" columns
    test_answer = pd.DataFrame({
        "id": range(1, df_local_test.shape[0] + 1),
        "hand": df_local_test["hand"].values
    })
    test_answer.to_csv(private / "test_answer.csv", index=False)
    
    # Create sample_submission.csv with dummy "hand" values (all 0)
    sample_submission = pd.DataFrame({
        "id": range(1, df_local_test.shape[0] + 1),
        "hand": [0] * df_local_test.shape[0]
    })
    sample_submission.to_csv(public / "sample_submission.csv", index=False)
    
    print("Data reorganization completed successfully.")

    # Clean up
    shutil.rmtree(raw)