#!/usr/bin/env python3
import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

def prepare(raw: Path, public: Path, private: Path):
    """
    Prepare data for the shopee-product-matching competition.
    
    Args:
        raw: Path to raw data directory
        public: Path to public data directory
        private: Path to private data directory
    """
    # Set fixed random seed for reproducibility
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    
    # Create destination directories
    public_train_dir = public / "train"
    public_test_dir = public / "test"
    public_train_images_dir = public / "train_images"
    
    for d in [public_train_dir, public_test_dir, public_train_images_dir, private]:
        d.mkdir(parents=True, exist_ok=True)
    
    # 1. Read original train.csv and test.csv
    print("Reading original CSV files...")
    train_df = pd.read_csv(raw / "train.csv")
    test_orig_df = pd.read_csv(raw / "test.csv")
    total_train_rows = len(train_df)
    holdout_size = max(len(test_orig_df), total_train_rows // 5)
    print(f"Total rows in original train.csv: {total_train_rows}")
    print(f"Holdout size (rows in original test.csv): {holdout_size}")

    # 2. Split train_df into new train and pseudo test sets
    test_df = train_df.sample(n=holdout_size, random_state=RANDOM_SEED)
    train_remaining_df = train_df.drop(test_df.index)
    
    # Assertions
    assert len(train_remaining_df) + len(test_df) == total_train_rows, "Split row counts don't sum to original count."
    assert len(test_df) == holdout_size, "Test split size != holdout_size from original test.csv."
    assert list(train_remaining_df.columns) == list(train_df.columns), "Columns of new train set differ."
    assert list(test_df.columns) == list(train_df.columns), "Columns of new test set differ."

    # 3. Write new training set and pseudo test set
    print("Writing new train.csv and test.csv...")
    train_remaining_df.to_csv(public_train_dir / "train.csv", index=False)
    test_df.drop(columns=['label_group']).to_csv(public_test_dir / "test.csv", index=False)
    
    # 4. Create ground truth file and sample_submission.csv
    print("Constructing test_answer.csv based on the pseudo test set...")
    test_answer = pd.DataFrame()
    test_answer["posting_id"] = test_df["posting_id"]
    
    # Build matches from label_group groups
    label_to_matches = {}
    for label_group, group in test_df.groupby("label_group"):
        posting_ids = sorted(group["posting_id"].astype(str).tolist())
        match_str = " ".join(posting_ids)
        label_to_matches[label_group] = match_str

    test_answer["matches"] = test_df["label_group"].map(label_to_matches)
    test_answer.to_csv(private / "test_answer.csv", index=False)
    
    # Create sample_submission.csv with dummy answers (self-match only)
    print("Generating sample_submission.csv with dummy answers...")
    sample_submission = pd.DataFrame()
    sample_submission["posting_id"] = test_answer["posting_id"]
    sample_submission["matches"] = sample_submission["posting_id"].astype(str)
    sample_submission.to_csv(public / "sample_submission.csv", index=False)
    
    # Assertions for test_answer.csv and sample_submission.csv
    for file_path in [private / "test_answer.csv", public / "sample_submission.csv"]:
        df = pd.read_csv(file_path)
        assert list(df.columns) == ["posting_id", "matches"], f"File {file_path} does not have required columns."
        assert len(df) == holdout_size, f"File {file_path} has wrong number of rows."
    
    # 5. Copy train_images folder
    print("Copying train_images folder...")
    src_train_images = raw / "train_images"
    
    # If destination folder already exists, remove it first
    if public_train_images_dir.exists():
        shutil.rmtree(public_train_images_dir)
    
    # Count total images for verification
    image_files = []
    for root, _, files in os.walk(src_train_images):
        for f in files:
            image_files.append(os.path.join(root, f))
    total_images = len(image_files)
    
    # Copy the directory structure and files with progress bar
    for root, dirs, files in os.walk(src_train_images):
        rel_path = os.path.relpath(root, src_train_images)
        dst_dir = public_train_images_dir / rel_path
        dst_dir.mkdir(exist_ok=True, parents=True)
        for f in tqdm(files, desc=f"Copying images from {root}"):
            src_file = Path(root) / f
            dst_file = dst_dir / f
            shutil.copy2(src_file, dst_file)
            
    # Verify total image count
    copied_image_files = []
    for root, _, files in os.walk(public_train_images_dir):
        for f in files:
            copied_image_files.append(os.path.join(root, f))
    assert len(copied_image_files) == total_images, "Mismatch in number of images copied."
    
    # Final assertions
    new_train = pd.read_csv(public_train_dir / "train.csv")
    new_test = pd.read_csv(public_test_dir / "test.csv")
    assert len(new_train) + len(new_test) == total_train_rows, "Final row count mismatch compared to original train.csv."
    
    test_answer_df = pd.read_csv(private / "test_answer.csv")
    sample_submission_df = pd.read_csv(public / "sample_submission.csv")
    assert list(test_answer_df.columns) == list(sample_submission_df.columns), "Column headers do not match between test_answer.csv and sample_submission.csv."
    assert len(test_answer_df) == len(sample_submission_df), "Row count mismatch between test_answer.csv and sample_submission.csv."
    
    print("Data reorganization completed successfully.")

    shutil.rmtree(raw)