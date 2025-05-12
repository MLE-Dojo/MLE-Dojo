#!/usr/bin/env python3
import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def prepare(raw: Path, public: Path, private: Path):
    """
    Prepares data for the quora-insincere-questions-classification competition by
    splitting raw data into public and private datasets.
    
    Args:
        raw: Path to the raw data directory
        public: Path to the public output directory
        private: Path to the private output directory
    """
    os.system(f"rm quora-insincere-questions-classification.zip")
    os.system("unzip embeddings.zip")
    os.system("rm embeddings.zip")
    # Create output directories
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    
    # Read original CSV files
    train_df = pd.read_csv(raw / "train.csv")
    orig_test_df = pd.read_csv(raw / "test.csv")
    sample_sub_orig_path = raw / "sample_submission.csv"
    
    # Determine split ratio: r = N_origTest / (N_train + N_origTest)
    N_train = len(train_df)
    N_orig_test = len(orig_test_df)
    r = N_orig_test / (N_train + N_orig_test)
    
    # Perform stratified train test split on train.csv using fixed random seed
    local_train, local_test = train_test_split(
        train_df,
        test_size=r,
        stratify=train_df["target"],
        random_state=42
    )
    
    # Save local training set with answers to public/train.csv
    local_train.to_csv(public / "train.csv", index=False)
    
    # Save local test set with answers to public/test.csv
    local_test.to_csv(public / "test.csv", index=False)
    
    # Build sample_submission.csv for the local test set
    sample_submission = pd.DataFrame({
        "qid": local_test["qid"],
        "prediction": 0  # dummy integer prediction for each row
    })
    sample_submission["prediction"] = sample_submission["prediction"].astype(int)
    sample_submission.to_csv(public / "sample_submission.csv", index=False)
    
    # Save test answers (qid and target) to private/test_answer.csv
    test_answer = local_test[["qid", "target"]].copy()
    test_answer.to_csv(private / "test_answer.csv", index=False)
    
    # Copy the embedding folders and original files
    embedding_dirs = [
        "GoogleNews-vectors-negative300",
        "glove.840B.300d",
        "paragram_300_sl999",
        "wiki-news-300d-1M"
    ]
    
    for d in embedding_dirs:
        src_dir = raw / d
        dest_dir = public / d
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(src_dir, dest_dir)
    
    
    # Verify data integrity
    _verify_data_integrity(train_df, local_train, local_test, public, private, r)
    
    print("Data preparation completed successfully.")
    
    # Clean up
    shutil.rmtree(raw)


def _verify_data_integrity(train_df, local_train, local_test, public, private, expected_ratio):
    """Helper function to verify data integrity with assertions"""
    
    # Total Count Check
    total_split_rows = len(pd.read_csv(public / "train.csv")) + len(pd.read_csv(public / "test.csv"))
    assert total_split_rows == len(train_df), "Total row count after split does not match original train.csv."
    
    # Ratio Check
    local_ratio = len(local_test) / len(train_df)
    assert abs(local_ratio - expected_ratio) < 0.05, "Local test split ratio does not approximate the expected ratio."
    
    # Stratification Check
    orig_pct = train_df["target"].mean()
    train_pct = local_train["target"].mean()
    test_pct = local_test["target"].mean()
    assert abs(orig_pct - train_pct) < 0.02, "Train split target distribution deviates more than 2%."
    assert abs(orig_pct - test_pct) < 0.02, "Test split target distribution deviates more than 2%."
    
    # QID Consistency Check
    local_test_public = pd.read_csv(public / "test.csv")
    test_answer_private = pd.read_csv(private / "test_answer.csv")
    assert local_test_public["qid"].tolist() == test_answer_private["qid"].tolist(), "QID order mismatches between public test.csv and private test_answer.csv."
    
    # Check sample_submission.csv format
    sample_sub_df = pd.read_csv(public / "sample_submission.csv")
    assert list(sample_sub_df.columns) == ["qid", "prediction"], "sample_submission.csv headers mismatch."
    assert sample_sub_df["prediction"].dtype == int or np.issubdtype(sample_sub_df["prediction"].dtype, np.integer), "Prediction column is not of integer type."
    assert len(sample_sub_df) == len(local_test_public), "Number of rows in sample_submission.csv does not match local test set."
    
    # File Integrity Check for embedding assets
    embedding_dirs = [
        "GoogleNews-vectors-negative300",
        "glove.840B.300d",
        "paragram_300_sl999",
        "wiki-news-300d-1M"
    ]
    
    for d in embedding_dirs:
        dest_dir = public / d
        assert dest_dir.exists(), f"Copied folder {d} not found in public directory."
    