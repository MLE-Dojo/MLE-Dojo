import os
import pandas as pd
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

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
    
    # Create target directories if they don't exist
    os.makedirs(public, exist_ok=True)
    os.makedirs(private, exist_ok=True)
    print(f"Ensured target directories exist: {public} and {private}")
    
    # Define file paths for raw data
    train_file = raw / "train.csv"
    test_file = raw / "test.csv"
    
    # Load raw train and test files
    print("Loading train.csv and test.csv from raw data...")
    try:
        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(test_file)
        print(f"Loaded train.csv with {len(df_train)} rows and test.csv with {len(df_test)} rows.")
    except Exception as e:
        print("Error loading raw data files:", e)
        return
    
    # Calculate split ratio based on original test and train sizes
    try:
        ratio = len(df_test) / len(df_train)
        if ratio > 1 or ratio < 0.05:
            print(f"Calculated ratio {ratio:.4f} is improper. Using default ratio 0.2")
            ratio = 0.2
        else:
            print(f"Using calculated split ratio: {ratio:.4f}")
    except Exception as e:
        print("Error calculating split ratio:", e)
        ratio = 0.2
    
    print(f"Splitting original train data into new train and test sets using test_size = {ratio:.4f}...")
    new_train, new_test = train_test_split(df_train, test_size=ratio, random_state=42)
    print(f"New train set size: {len(new_train)} rows, New test set size: {len(new_test)} rows")
    
    # Validate that the split is proper
    assert len(new_train) + len(new_test) == len(df_train), "Error: The train/test split does not sum to original data size."
    print("Train/test split validation passed.")
    
    # Prepare new test files for evaluation:
    # Reset index to create a row_id column for the submission files
    new_test = new_test.reset_index(drop=True)
    # Create sample_submission with only row_id and meter_reading (agent's predictions will go here)
    sample_submission = pd.DataFrame({
        "row_id": new_test.index,
        "meter_reading": new_test["meter_reading"]
    })
    # test_answer will contain the true meter_reading values with the same structure as sample_submission
    test_answer = sample_submission.copy()
    
    # Save new train set to public folder
    new_train_path = public / "train.csv"
    new_test_public = new_test.drop(columns=["meter_reading"])  # Remove target column for public test file
    new_test_path = public / "test.csv"
    sample_submission_path = public / "sample_submission.csv"
    test_answer_path = private / "test_answer.csv"
    
    print("Saving new train and test files to public folder...")
    new_train.to_csv(new_train_path, index=False)
    new_test_public.to_csv(new_test_path, index=False)
    sample_submission.to_csv(sample_submission_path, index=False)
    print(f"Saved new train file to {new_train_path}")
    print(f"Saved new test file (without targets) to {new_test_path}")
    print(f"Saved sample_submission file to {sample_submission_path}")
    
    # Save test_answer file (with true labels) to private folder
    test_answer.to_csv(test_answer_path, index=False)
    print(f"Saved test_answer file to {test_answer_path}")
    
    # Validation: Check if sample_submission and test_answer have the same columns
    cols_submission = set(sample_submission.columns)
    cols_answer = set(test_answer.columns)
    assert cols_submission == cols_answer, "Error: Columns in sample_submission and test_answer.csv do not match."
    print("Validation passed: sample_submission and test_answer.csv have matching columns.")
    
    # Copy additional files (if any) from raw folder to public folder
    additional_files = ["building_metadata.csv", "weather_train.csv", "weather_test.csv"]
    print("Copying additional files to public folder...")
    for file in additional_files:
        src = raw / file
        dest = public / file
        if os.path.exists(src):
            shutil.copy(src, dest)
            print(f"Copied {file} to {public}")
        else:
            print(f"Warning: {file} not found in raw directory.")
    
    print("Data preparation complete.")

    # clean up
    shutil.rmtree(raw)