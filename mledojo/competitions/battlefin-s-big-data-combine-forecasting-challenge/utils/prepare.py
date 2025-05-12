import os
import shutil
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def prepare(raw: Path, public: Path, private: Path):
    """
    Prepare data for the competition by splitting train/test data and organizing files.
    
    Args:
        raw: Path to the raw data directory
        public: Path to the public data directory
        private: Path to the private data directory
    """
    print("Starting data preparation...")

    # Create target directories if they do not exist
    for directory in [public, private]:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Directory exists or created: {directory}")

    # Load trainLabels.csv from the raw directory
    train_labels_path = raw / "trainLabels.csv"
    try:
        df_labels = pd.read_csv(train_labels_path)
        print(f"Loaded train labels from {train_labels_path} with shape {df_labels.shape}")
    except Exception as e:
        print("Error loading trainLabels.csv:", e)
        return

    # Calculate split ratio based on original test/train sizes
    # Original sizes: 200 training days and 310 testing days -> ratio = 310/200 = 1.55 (invalid)
    orig_train_size = 200
    orig_test_size = 310
    calculated_ratio = orig_test_size / orig_train_size  # 1.55
    default_ratio = 0.2  # Default test ratio if calculated ratio is invalid

    if calculated_ratio > 1 or calculated_ratio < 0.05:
        print(f"Calculated ratio {calculated_ratio} is invalid. Using default ratio {default_ratio}.")
        test_ratio = default_ratio
    else:
        test_ratio = calculated_ratio

    total_samples = df_labels.shape[0]
    expected_test_size = int(total_samples * test_ratio)
    expected_train_size = total_samples - expected_test_size
    print(f"Splitting data into approximately {expected_train_size} training samples and {expected_test_size} testing samples.")

    # Perform train-test split
    train_df, test_df = train_test_split(df_labels, test_size=test_ratio, random_state=42)
    print(f"After splitting: train shape = {train_df.shape}, test shape = {test_df.shape}")

    # Validate that there is no overlap in FileIds between train and test
    overlap = set(train_df["FileId"]).intersection(set(test_df["FileId"]))
    assert len(overlap) == 0, "Error: Overlapping FileIds found between training and testing splits!"
    print("Validation passed: No overlapping FileIds between train and test sets.")

    # Save the new train and test splits to the public directory
    train_out_path = public / "train.csv"
    test_out_path = public / "test.csv"
    train_df.to_csv(train_out_path, index=False)
    test_df.to_csv(test_out_path, index=False)
    print(f"Saved training data to {train_out_path} and testing data to {test_out_path}")

    # Create test_answer.csv in the private directory (contains true outputs for the test split)
    test_answer_path = private / "test_answer.csv"
    test_df.to_csv(test_answer_path, index=False)
    print(f"Saved test answer data to {test_answer_path}")

    # Create sample_submission.csv in the public directory
    # It should have the same FileId and output columns as test_answer.csv but with zero predictions.
    sample_submission_df = test_df.copy()
    output_columns = [col for col in sample_submission_df.columns if col != "FileId"]
    sample_submission_df[output_columns] = 0
    sample_submission_path = public / "sample_submission.csv"
    sample_submission_df.to_csv(sample_submission_path, index=False)
    print(f"Saved sample submission data to {sample_submission_path}")

    # Validate that test_answer.csv and sample_submission.csv have the same columns
    test_answer_df = pd.read_csv(test_answer_path)
    sample_submission_df_check = pd.read_csv(sample_submission_path)
    assert list(test_answer_df.columns) == list(sample_submission_df_check.columns), "Error: Columns mismatch between test_answer.csv and sample_submission.csv!"
    print("Validation passed: test_answer.csv and sample_submission.csv have matching columns.")

    # Copy necessary data files from the raw data folder to the public directory
    raw_data_dir = raw / "data"
    public_data_dir = public / "data"
    if raw_data_dir.exists():
        public_data_dir.mkdir(exist_ok=True)
        print(f"Created or verified public data directory: {public_data_dir}")
        
        # Only copy files that are needed for the new train/test split
        train_file_ids = set(train_df["FileId"])
        test_file_ids = set(test_df["FileId"])
        needed_file_ids = train_file_ids.union(test_file_ids)
        
        for file_path in raw_data_dir.iterdir():
            if file_path.is_file():
                file_id = file_path.stem
                try:
                    if int(file_id) in needed_file_ids:
                        dst_file = public_data_dir / file_path.name
                        shutil.copy(file_path, dst_file)
                        print(f"Copied {file_path} to {dst_file}")
                except ValueError:
                    # If filename is not an integer, skip it
                    pass
    else:
        print("No additional raw data directory found to copy.")

    print("Data preparation completed successfully.")

    # clean up
    shutil.rmtree(raw)

