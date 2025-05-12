import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

def prepare(raw: Path, public: Path, private: Path):
    """
    Prepares data for the linking-writing-processes-to-writing-quality competition by
    splitting raw data into public and private datasets.
    
    Args:
        raw: Path to the raw data directory
        public: Path to the public output directory
        private: Path to the private output directory
    """
    # Create output directories
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories in public
    (public / "train").mkdir(exist_ok=True)
    (public / "test").mkdir(exist_ok=True)
    
    # Set constants
    TRAIN_RATIO = 0.67  # roughly 67% for training, 33% for test
    RANDOM_STATE = 42  # for reproducibility
    
    # Read original files
    df_scores = pd.read_csv(raw / "train_scores.csv")
    df_logs = pd.read_csv(raw / "train_logs.csv")
    
    # Get set of unique essay IDs from train_scores.csv
    score_ids = set(df_scores["id"].unique())
    log_ids = set(df_logs["id"].unique())
    
    # Assertion: All IDs in train_scores.csv are in train_logs.csv
    assert score_ids.issubset(log_ids), "Not all essay IDs in train_scores.csv are present in train_logs.csv."
    
    # Split the scores based on unique essay IDs
    unique_ids = list(df_scores["id"].unique())
    split_train_ids, split_test_ids = train_test_split(unique_ids, train_size=TRAIN_RATIO, random_state=RANDOM_STATE)
    
    # Create the split dataframes
    df_train_scores = df_scores[df_scores["id"].isin(split_train_ids)]
    df_test_scores = df_scores[df_scores["id"].isin(split_test_ids)]  # this will be our test answer file
    
    df_train_logs = df_logs[df_logs["id"].isin(split_train_ids)]
    df_test_logs = df_logs[df_logs["id"].isin(split_test_ids)]
    
    # Final assertions: The union of the IDs should equal the full set from train_scores.csv
    combined_ids = set(df_train_scores["id"].unique()).union(set(df_test_scores["id"].unique()))
    assert combined_ids == score_ids, "The union of the split IDs does not match the original set of IDs."
    
    # Write new files
    # Training set: logs and scores
    df_train_logs.to_csv(public / "train" / "train_logs.csv", index=False)
    df_train_scores.to_csv(public / "train" / "train_scores.csv", index=False)
    
    # Test set: logs and answers
    df_test_logs.to_csv(public / "test" / "test_logs.csv", index=False)
    df_test_scores.to_csv(private / "test_answer.csv", index=False)
    
    # Build a new sample_submission.csv for the new test set based on test_answer.csv
    df_sample_submission = df_test_scores.copy()
    df_sample_submission["score"] = 1.0  # Using a placeholder dummy value of type float
    df_sample_submission.to_csv(public / "sample_submission.csv", index=False)
    
    # Verify sample_submission.csv
    assert df_sample_submission.shape[0] == df_test_scores.shape[0], "Row count mismatch in sample_submission.csv."
    
    # Log the counts of unique essay IDs
    print("Final counts of unique essay IDs:")
    print(f"Total in original train_scores.csv: {len(score_ids)}")
    print(f"Training set IDs: {len(set(df_train_scores['id'].unique()))}")
    print(f"Test set IDs: {len(set(df_test_scores['id'].unique()))}")
    
    print("Data preparation completed successfully.")

    # Clean up
    shutil.rmtree(raw)