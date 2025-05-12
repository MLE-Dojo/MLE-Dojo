import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import shutil

def prepare(raw: Path, public: Path, private: Path):
    """
    Prepare data by splitting original training data into new train and test sets.
    
    Args:
        raw: Path to the directory containing raw data files
        public: Path to the directory where public files will be saved
        private: Path to the directory where private files will be saved
    """
    # Print starting message
    print("Starting data preparation process...")

    # Create output directories if they don't exist
    print("Creating output directories...")
    public.mkdir(exist_ok=True, parents=True)
    private.mkdir(exist_ok=True, parents=True)

    # Load the original data
    print("Loading original data...")
    train_data = pd.read_csv(raw / "train.csv")
    test_data = pd.read_csv(raw / "test.csv")
    orig_sample_submission = pd.read_csv(raw / "sampleSubmission.csv")

    # Calculate the split ratio based on original sizes
    print("Calculating split ratio...")
    try:
        original_train_size = len(train_data)
        original_test_size = len(test_data)
        split_ratio = original_test_size / (original_train_size + original_test_size)

        # Check if the ratio is reasonable
        if split_ratio <= 0.05 or split_ratio >= 1:
            print(f"Calculated ratio {split_ratio} is outside reasonable bounds. Using default ratio of 0.2")
            split_ratio = 0.2
        else:
            print(f"Using calculated split ratio: {split_ratio}")
    except Exception as e:
        print(f"Error calculating split ratio: {e}. Using default ratio of 0.2")
        split_ratio = 0.2

    print(f"Original train size: {original_train_size}, Original test size: {original_test_size}")
    print(f"Split ratio: {split_ratio}")

    # Split the training data into new train and test sets
    print("Splitting original training data...")
    new_train, new_test = train_test_split(train_data, test_size=split_ratio, random_state=42)

    print(f"New train size: {len(new_train)}, New test size: {len(new_test)}")

    # Get the label columns (all columns present in both the original sample submission and train data)
    label_columns = [col for col in orig_sample_submission.columns if col in train_data.columns and col != 'id']
    print(f"Label columns: {label_columns}")

    # Create test_answer.csv (contains the true labels for the new test set)
    print("Creating test_answer.csv...")
    test_answer = new_test[['id'] + label_columns].copy()

    # Create sample_submission.csv (contains the format for submissions)
    print("Creating sample_submission.csv...")
    sample_submission = pd.DataFrame(0, index=range(len(new_test)), columns=['id'] + label_columns)
    sample_submission['id'] = new_test['id'].values

    # Save the files
    print("Saving new files...")
    new_train.to_csv(public / "train.csv", index=False, quoting=1)  # quoting=1 for quoting all fields

    # For the test file, we only include id, tweet, state, location (similar to original test.csv)
    new_test[['id', 'tweet', 'state', 'location']].to_csv(public / "test.csv", index=False, quoting=1)

    sample_submission.to_csv(public / "sample_submission.csv", index=False)
    test_answer.to_csv(private / "test_answer.csv", index=False)

    # Copy additional files if they exist
    print("Checking for additional files...")
    try:
        variable_names_path = raw / "variableNames.txt"
        if variable_names_path.exists():
            # Try to read as CSV first
            try:
                variable_names = pd.read_csv(variable_names_path)
                variable_names.to_csv(public / "variableNames.txt", index=False)
                print("Copied variableNames.txt as CSV")
            except:
                # If that fails, read as plain text
                with open(variable_names_path, 'r') as f_in:
                    content = f_in.read()
                with open(public / "variableNames.txt", 'w') as f_out:
                    f_out.write(content)
                print("Copied variableNames.txt as text")
    except Exception as e:
        print(f"Error handling additional files: {e}")

    # Validations
    print("Running validations...")

    # Check if the test train split is proper
    train_test_ratio = len(new_test) / (len(new_train) + len(new_test))
    assert abs(
        train_test_ratio - split_ratio) < 0.01, f"Test-train split ratio {train_test_ratio} does not match expected {split_ratio}"
    print(f"Test-train split validation passed: {train_test_ratio} ≈ {split_ratio}")

    # Check if test_answer.csv and sample_submission.csv have the same columns
    test_answer_cols = pd.read_csv(private / "test_answer.csv").columns.tolist()
    sample_submission_cols = pd.read_csv(public / "sample_submission.csv").columns.tolist()
    assert test_answer_cols == sample_submission_cols, f"Column mismatch: {test_answer_cols} != {sample_submission_cols}"
    print("Column validation passed: test_answer.csv and sample_submission.csv have the same columns")

    print("Data preparation completed successfully!")

    # clean up
    shutil.rmtree(raw)
