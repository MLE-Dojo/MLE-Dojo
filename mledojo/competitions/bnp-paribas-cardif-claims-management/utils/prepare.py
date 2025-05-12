import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def prepare(raw: Path, public: Path, private: Path):
    """
    Prepare data by splitting raw data into public and private datasets.
    
    Args:
        raw: Path to raw data directory
        public: Path to public data directory
        private: Path to private data directory
    """
    print("Starting data preparation...")

    # Create target directories if they do not exist
    os.makedirs(public, exist_ok=True)
    os.makedirs(private, exist_ok=True)
    print(f"Ensured directories exist: {public} and {private}")

    # unzip the zip files
    os.system(f"unzip {raw / 'train.csv.zip'} -d {raw}")
    os.system(f"unzip {raw / 'test.csv.zip'} -d {raw}")
    os.system(f"unzip {raw / 'sample_submission.csv.zip'} -d {raw}")

    # Define file paths
    train_path = raw / "train.csv"
    test_path = raw / "test.csv"
    sample_submission_path = raw / "sample_submission.csv"

    # Read input files
    print("Reading train.csv ...")
    df_train = pd.read_csv(train_path)
    print(f"Train data shape: {df_train.shape}")

    print("Reading test.csv ...")
    df_test_orig = pd.read_csv(test_path)
    print(f"Original test data shape: {df_test_orig.shape}")

    print("Reading sample_submission.csv ...")
    df_sample_submission = pd.read_csv(sample_submission_path)
    print(f"Sample submission shape: {df_sample_submission.shape}")

    # Calculate split ratio based on original test/train sizes
    try:
        ratio = len(df_test_orig) / len(df_train)
        if ratio > 1 or ratio < 0.05:
            print(f"Calculated ratio ({ratio}) is improper. Using default ratio of 0.2")
            ratio = 0.2
        else:
            print(f"Calculated split ratio: {ratio}")
    except Exception as e:
        print(f"Error calculating ratio: {e}. Using default ratio of 0.2")
        ratio = 0.2

    # Split the original training data into new train and test sets
    print("Splitting train data into new train and test splits ...")
    # Stratify based on 'target' to maintain distribution
    new_train, new_test = train_test_split(df_train, test_size=ratio, random_state=42, stratify=df_train['target'])
    print(f"New train shape: {new_train.shape}, New test shape: {new_test.shape}")

    # Validation: check if split sizes add up correctly
    assert len(new_train) + len(new_test) == len(df_train), "Train/Test split sizes do not add up to original train size!"
    print("Validation passed: Train/Test split sizes add up to original train size.")

    # Save new train and test splits to public folder
    new_train_path = public / "train.csv"
    new_test_path = public / "test.csv"
    print(f"Saving new train data to {new_train_path}")
    new_train.to_csv(new_train_path, index=False)
    print(f"Saving new test data to {new_test_path}")
    new_test.to_csv(new_test_path, index=False)

    # Create test_answer.csv from new_test with the ground truth answers.
    # Rename the 'target' column to 'PredictedProb' and retain the 'ID'
    if 'target' in new_test.columns:
        test_answer = new_test[['ID', 'target']].copy()
        test_answer.rename(columns={'target': 'PredictedProb'}, inplace=True)
        print("Created test_answer dataframe with columns:", list(test_answer.columns))
    else:
        raise Exception("Column 'target' not found in new test data!")

    # Create sample_submission.csv with the same IDs and dummy predictions (0.5)
    sample_sub = test_answer[['ID']].copy()
    sample_sub['PredictedProb'] = 0.5
    print("Created sample_submission dataframe with columns:", list(sample_sub.columns))

    # Validate that test_answer.csv and sample_submission.csv have the same columns
    assert list(test_answer.columns) == list(sample_sub.columns), "Columns of test_answer and sample_submission do not match!"
    print("Validation passed: test_answer.csv and sample_submission.csv have matching columns.")

    # Save test_answer.csv to private folder and sample_submission.csv to public folder
    test_answer_path = private / "test_answer.csv"
    sample_submission_new_path = public / "sample_submission.csv"
    print(f"Saving test_answer data to {test_answer_path}")
    test_answer.to_csv(test_answer_path, index=False)
    print(f"Saving sample_submission data to {sample_submission_new_path}")
    sample_sub.to_csv(sample_submission_new_path, index=False)


    print("Data preparation completed successfully.")

    # clean up
    shutil.rmtree(raw)
