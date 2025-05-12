import os
import pandas as pd
import shutil
from pathlib import Path

def prepare(raw: Path, public: Path, private: Path):
    """
    Prepares data for evaluation by:
    1. Loading the original data
    2. Creating a new train/test split from the training data
    3. Creating test_answer.csv and sample_submission.csv
    4. Saving all files to the appropriate directories
    
    Args:
        raw: Path to the raw data directory
        public: Path to the public output directory
        private: Path to the private output directory
    """
    print("Starting data preparation...")
    
    # Create output directories if they don't exist
    os.makedirs(public, exist_ok=True)
    print(f"Created directory: {public}")
    os.makedirs(private, exist_ok=True)
    print(f"Created directory: {private}")
    
    # unzip the gz files
    os.system(f"gunzip -c {raw / 'train.gz'} > {raw / 'train.csv'}")
    os.system(f"gunzip -c {raw / 'test.gz'} > {raw / 'test.csv'}")
    os.system(f"gunzip -c {raw / 'sampleSubmission.gz'} > {raw / 'sampleSubmission.csv'}")

    # Define file paths for raw files
    train_file = raw / "train.csv"
    test_file = raw / "test.csv"
    sample_submission_file = raw / "sampleSubmission.csv"
    
    # Load data files
    try:
        df_train = pd.read_csv(train_file)
        print(f"Loaded training data from {train_file} with {len(df_train)} rows.")
    except Exception as e:
        print("Error loading training file:", e)
        return
        
    try:
        df_test = pd.read_csv(test_file)
        print(f"Loaded test data from {test_file} with {len(df_test)} rows.")
    except Exception as e:
        print("Error loading test file:", e)
        return
        
    try:
        df_sample_submission = pd.read_csv(sample_submission_file)
        print(f"Loaded sample submission data from {sample_submission_file} with {len(df_sample_submission)} rows.")
    except Exception as e:
        print("Error loading sample submission file:", e)
        return
        
    # Calculate split ratio based on original test/train sizes
    try:
        ratio = len(df_test) / len(df_train)
        if ratio > 1 or ratio < 0.05:
            print(f"Calculated ratio ({ratio:.4f}) is improper. Using default ratio 0.1.")
            ratio = 0.1
        else:
            print(f"Using calculated split ratio: {ratio:.4f}")
    except Exception as e:
        print("Error calculating split ratio:", e)
        ratio = 0.1
        print("Using default ratio: 0.1")
    
    # Split the training data into new train and test sets
    new_test = df_train.sample(frac=ratio, random_state=42)
    new_train = df_train.drop(new_test.index)
    print(f"Split training data into new train ({len(new_train)} rows) and new test ({len(new_test)} rows).")
    
    # Validation: check if split sizes add up to original training size
    assert len(new_train) + len(new_test) == len(df_train), "Train/Test split sizes do not add up to original training size."
    print("Validation passed: Train and test split sizes add up correctly.")
    
    # Define output file paths
    public_train_file = public / "train.csv"
    public_test_file = public / "test.csv"
    private_test_answer_file = private / "test_answer.csv"
    public_sample_submission_file = public / "sample_submission.csv"
    
    # Save new training data to public directory
    new_train.to_csv(public_train_file, index=False)
    print(f"Saved new training data to {public_train_file}")
    
    # Create public test file without the target column 'click'
    if 'click' in new_test.columns:
        new_test_public = new_test.drop(columns=['click'])
        print("Removed 'click' column from new test data for public release.")
    else:
        new_test_public = new_test.copy()
        print("No 'click' column found in new test data.")
    new_test_public.to_csv(public_test_file, index=False)
    print(f"Saved new public test data to {public_test_file}")
    
    # Create test_answer.csv with 'id' and 'click' columns
    if {'id', 'click'}.issubset(new_test.columns):
        test_answer = new_test[['id', 'click']]
    else:
        print("Error: 'id' and/or 'click' columns not found in new test data.")
        return
    test_answer.to_csv(private_test_answer_file, index=False)
    print(f"Saved test answers to {private_test_answer_file}")
    
    # Create sample_submission.csv with same columns as test_answer and dummy predictions (e.g., 0.5)
    sample_submission = test_answer.copy()
    sample_submission['click'] = 0.5
    sample_submission.to_csv(public_sample_submission_file, index=False)
    print(f"Saved sample submission to {public_sample_submission_file}")
    
    # Validate that test_answer.csv and sample_submission.csv have the same columns
    assert list(test_answer.columns) == list(sample_submission.columns), "Columns of test_answer and sample_submission do not match."
    print("Validation passed: test_answer and sample_submission columns match.")
    
    # # Copy additional files (if any) from raw directory to public directory
    # raw_files = os.listdir(raw)
    # for file in raw_files:
    #     if file not in ["train", "test", "sampleSubmission"]:
    #         src = raw / file
    #         dst = public / file
    #         try:
    #             shutil.copy(src, dst)
    #             print(f"Copied additional file '{file}' to public directory.")
    #         except Exception as e:
    #             print(f"Error copying file '{file}':", e)
    
    # print("Data preparation complete.")

    # clean up
    shutil.rmtree(raw)
