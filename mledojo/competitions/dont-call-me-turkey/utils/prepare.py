import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

def prepare(raw: Path, public: Path, private: Path):
    # Print starting message
    print("Starting data preparation process...")

    # Create output directories if they don't exist
    print("Creating output directories...")
    os.makedirs(public, exist_ok=True)
    os.makedirs(private, exist_ok=True)

    # Load the original data
    print("Loading original data...")
    with open(raw / "train.json", "r") as f:
        train_data = json.load(f)

    with open(raw / "test.json", "r") as f:
        test_data = json.load(f)

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

    # Create test_answer.csv (contains the true labels for the new test set)
    print("Creating test_answer.csv...")
    test_answer = pd.DataFrame([{"vid_id": item["vid_id"], "is_turkey": item["is_turkey"]} for item in new_test])

    # Create sample_submission.csv (contains the format for submissions)
    print("Creating sample_submission.csv...")
    sample_submission = pd.DataFrame([{"vid_id": item["vid_id"], "is_turkey": 0} for item in new_test])

    # Save the files
    print("Saving new files...")
    with open(public / "train.json", "w") as f:
        json.dump(new_train, f)

    with open(public / "test.json", "w") as f:
        # For the test file, we remove the is_turkey field (similar to original test.json)
        new_test_without_labels = []
        for item in new_test:
            item_copy = item.copy()
            if "is_turkey" in item_copy:
                item_copy.pop("is_turkey")
            new_test_without_labels.append(item_copy)
        json.dump(new_test_without_labels, f)

    sample_submission.to_csv(public / "sample_submission.csv", index=False)
    test_answer.to_csv(private / "test_answer.csv", index=False)

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

