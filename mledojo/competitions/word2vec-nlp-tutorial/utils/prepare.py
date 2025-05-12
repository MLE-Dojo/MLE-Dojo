#!/usr/bin/env python3
import os
import shutil
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def prepare(raw: Path, public: Path, private: Path):
    """
    Prepare data for word2vec-nlp-tutorial competition.
    
    Args:
        raw: Path to raw data directory
        public: Path to public data directory
        private: Path to private data directory
    """
    os.system("unzip -o " + str(raw / "labeledTrainData.tsv.zip") + " -d " + str(raw))
    os.system("unzip -o " + str(raw / "unlabeledTrainData.tsv.zip") + " -d " + str(raw))
    os.system("unzip -o " + str(raw / "testData.tsv.zip") + " -d " + str(raw))

    
    # 1. Read labeledTrainData.tsv
    labeled_train_path = raw / "labeledTrainData.tsv"
    df = pd.read_csv(labeled_train_path, sep="\t")
    
    # Assertion: Check original row count
    assert df.shape[0] == 25000, f"Expected 25000 rows, but got {df.shape[0]}"
    
    # 2. Perform stratified random split (50:50)
    train_df, test_df = train_test_split(
        df,
        test_size=0.5,
        stratify=df['sentiment'],
        random_state=42
    )
    
    # Assertions for split row counts and data integrity
    assert train_df.shape[0] == 12500, f"Training set should have 12500 rows, found {train_df.shape[0]}"
    assert test_df.shape[0] == 12500, f"Test set should have 12500 rows, found {test_df.shape[0]}"
    assert set(train_df['id']).isdisjoint(set(test_df['id'])), "Some ids appear in both training and test splits!"
    
    # 3. Write the splits to their respective files
    train_output_path = public / "labeled_train.tsv"
    test_output_path = public / "test.tsv"
    
    # Export training data with all columns
    train_df.to_csv(train_output_path, sep="\t", index=False)
    
    # Export test data without sentiment column
    test_df.drop(columns="sentiment").to_csv(test_output_path, sep="\t", index=False)
    
    # 4. Create test_answer.csv in private directory
    test_answer_df = test_df[['id', 'sentiment']]
    test_answer_path = private / "test_answer.csv"
    test_answer_df.to_csv(test_answer_path, index=False)
    
    # 5. Create sample_submission.csv in public directory
    sample_submission_path = public / "sample_submission.csv"
    dummy_submission = test_answer_df.copy()
    dummy_submission['sentiment'] = 0  # Set dummy sentiment values
    dummy_submission.to_csv(sample_submission_path, index=False)
    
    # 6. Copy unlabeledTrainData.tsv to public directory
    src_unlabeled_path = raw / "unlabeledTrainData.tsv"
    dst_unlabeled_path = public / "unlabeled_train.tsv"
    shutil.copy2(src_unlabeled_path, dst_unlabeled_path)
    
    # Final verification
    assert os.path.getsize(src_unlabeled_path) == os.path.getsize(dst_unlabeled_path), "File size mismatch after copy"

    # Clean up
    shutil.rmtree(raw)