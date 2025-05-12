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
    
    # Create output directories if they don't exist
    os.makedirs(public, exist_ok=True)
    os.makedirs(private, exist_ok=True)
    print(f"Ensured output directories: {public}, {private}")
    
    # Load datasets
    try:
        train_df = pd.read_csv(raw / "train.csv")
        test_df = pd.read_csv(raw / "test.csv")
        print(f"Loaded train.csv with shape {train_df.shape}")
        print(f"Loaded test.csv with shape {test_df.shape}")
    except Exception as e:
        print("Error loading CSV files:", e)
        return

    # Calculate split ratio based on original test/train sizes
    try:
        ratio = len(test_df) / len(train_df)
        if ratio > 1 or ratio < 0.05:
            print(f"Calculated ratio {ratio:.4f} is improper. Using default ratio 0.2 instead.")
            ratio = 0.2
        else:
            print(f"Calculated split ratio based on test/train sizes: {ratio:.4f}")
    except Exception as e:
        print("Error calculating split ratio:", e)
        ratio = 0.2

    # Split the train dataset into new train and test sets
    try:
        # Stratify on 'Activity' to maintain distribution
        new_train, new_test = train_test_split(train_df, test_size=ratio, random_state=42, stratify=train_df['Activity'])
        print(f"New train split shape: {new_train.shape}")
        print(f"New test split shape: {new_test.shape}")
        # Validate split sizes
        assert len(new_train) + len(new_test) == len(train_df), "Train-test split sizes do not add up!"
    except Exception as e:
        print("Error during train/test split:", e)
        return

    # For new test set, add an ID column so that sample_submission and test_answer have matching columns
    new_test = new_test.copy()
    new_test.insert(0, "MoleculeId", range(1, len(new_test) + 1))
    # Create test_answer (with labels) and sample_submission (without label, but same columns)
    test_answer = new_test[["MoleculeId", "Activity"]].copy()
    sample_submission = test_answer.copy()  # Ensuring both have exactly the same columns
    print("Created test_answer and sample_submission dataframes with matching columns.")
    
    # Save new train and test split files to public directory
    try:
        new_train.to_csv(public / "train.csv", index=False)
        # For public test file, drop the Activity column (simulate unseen labels)
        new_test.drop("Activity", axis=1).to_csv(public / "test.csv", index=False)
        print("Saved new train.csv and test.csv to public directory.")
    except Exception as e:
        print("Error saving new train/test files:", e)
        return

    # Save test_answer and sample_submission files
    try:
        test_answer.to_csv(private / "test_answer.csv", index=False)
        sample_submission.to_csv(public / "sample_submission.csv", index=False)
        print("Saved test_answer.csv to private and sample_submission.csv to public directory.")
    except Exception as e:
        print("Error saving test_answer or sample_submission files:", e)
        return

    # Validation: Check if test_answer.csv and sample_submission.csv have the same columns
    if list(test_answer.columns) == list(sample_submission.columns):
        print("Validation passed: test_answer and sample_submission have matching columns.")
    else:
        raise AssertionError("Columns of test_answer and sample_submission do not match!")
    
    print("Data preparation completed successfully.")

    # clean up
    shutil.rmtree(raw)
