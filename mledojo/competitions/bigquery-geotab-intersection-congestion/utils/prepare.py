import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
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

    # Define file paths
    train_file = raw / "train.csv"
    test_file = raw / "test.csv"
    sample_submission_file = raw / "sample_submission.csv"
    
    # Read original files
    print("Reading train.csv and test.csv...")
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    print(f"Original train shape: {df_train.shape}")
    print(f"Original test shape: {df_test.shape}")

    # Calculate split ratio based on original test/train sizes using Option B:
    # new_test_size = len(test.csv) / (len(train.csv) + len(test.csv))
    try:
        calculated_ratio = len(df_test) / (len(df_train) + len(df_test))
        if calculated_ratio > 1 or calculated_ratio < 0.05:
            print(f"Calculated ratio {calculated_ratio:.4f} is improper. Using default ratio 0.2.")
            test_ratio = 0.2
        else:
            test_ratio = calculated_ratio
        print(f"Using test split ratio: {test_ratio:.4f}")
    except Exception as e:
        print(f"Error in calculating ratio: {e}. Using default ratio 0.2")
        test_ratio = 0.2

    # Split the original train data into new train and test splits
    print("Splitting train.csv into new train and test sets...")
    new_train, new_test = train_test_split(df_train, test_size=test_ratio, random_state=42)
    print(f"New train shape: {new_train.shape}")
    print(f"New test shape: {new_test.shape}")

    # Validate split: ensure no overlap and total size is preserved
    assert set(new_train.index).isdisjoint(set(new_test.index)), "Error: Train and test splits overlap!"
    assert len(new_train) + len(new_test) == len(df_train), "Error: Train and test splits do not add up to original train size!"
    print("Train-test split validation passed.")

    # Save new train and test splits to public directory
    new_train_file = public / "train.csv"
    new_test_file = public / "test.csv"
    new_train.to_csv(new_train_file, index=False)
    new_test.to_csv(new_test_file, index=False)
    print(f"Saved new train.csv to {new_train_file}")
    print(f"Saved new test.csv to {new_test_file}")

    # Define the target columns needed for submission (6 targets)
    target_columns = [
        "TotalTimeStopped_p20", "TotalTimeStopped_p50", "TotalTimeStopped_p80",
        "DistanceToFirstStop_p20", "DistanceToFirstStop_p50", "DistanceToFirstStop_p80"
    ]
    for col in target_columns:
        if col not in new_test.columns:
            raise ValueError(f"Column {col} not found in new test split!")
    print("All required target columns found in new test split.")

    # Prepare test_answer.csv and sample_submission.csv in long format
    # For each row in new_test, create 6 rows with TargetId as "RowId_metricIndex"
    long_rows = []
    for idx, row in new_test.iterrows():
        base_id = row['RowId'] if 'RowId' in new_test.columns else idx
        for metric_idx, col in enumerate(target_columns):
            target_id = f"{base_id}_{metric_idx}"
            long_rows.append({"TargetId": target_id, "Target": row[col]})
    df_test_answer = pd.DataFrame(long_rows)
    print(f"Created test_answer DataFrame with shape: {df_test_answer.shape}")

    # Create sample_submission with the same TargetId values but default prediction 0
    df_sample_submission = df_test_answer.copy()
    df_sample_submission["Target"] = 0
    print("Created sample_submission DataFrame with default predictions.")

    # Validate that the columns of both files match
    assert list(df_test_answer.columns) == list(df_sample_submission.columns), "Error: Columns of test_answer and sample_submission do not match!"
    print("Validation passed: test_answer and sample_submission columns match.")

    # Save test_answer.csv to private directory and sample_submission.csv to public directory
    test_answer_path = private / "test_answer.csv"
    sample_submission_path = public / "sample_submission.csv"
    df_test_answer.to_csv(test_answer_path, index=False)
    df_sample_submission.to_csv(sample_submission_path, index=False)
    print(f"Saved test_answer.csv to {test_answer_path}")
    print(f"Saved sample_submission.csv to {sample_submission_path}")

    # Copy additional files from data_raw to public directory
    additional_files = ["submission_metric_map", "submission_metric_map.json"]
    for filename in additional_files:
        src_path = raw / filename
        dst_path = public / filename
        if os.path.exists(src_path):
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            else:
                shutil.copy(src_path, dst_path)
            print(f"Copied {filename} to {public}")
        else:
            print(f"{filename} not found in {raw}; skipping copy.")

    print("Data preparation completed.")

    # clean up
    shutil.rmtree(raw)
