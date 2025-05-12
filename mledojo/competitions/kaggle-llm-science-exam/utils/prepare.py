import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import subprocess

def prepare(raw: Path, public: Path, private: Path):
    """
    Prepares data for the Kaggle LLM Science Exam competition by splitting raw data
    into public and private datasets.
    
    Args:
        raw: Path to the raw data directory
        public: Path to the public output directory
        private: Path to the private output directory
    """
    print("Starting data preparation...")
    
    # Create output directories
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    
    # Setup paths
    original_dir = raw / "original"
    
    # Create original directory if it doesn't exist and move files
    if not original_dir.exists():
        print("Creating 'original' folder and moving files...")
        original_dir.mkdir(parents=True, exist_ok=True)
        
        for filename in ["train.csv", "test.csv", "sample_submission.csv"]:
            src_path = raw / filename
            if src_path.exists():
                shutil.move(str(src_path), str(original_dir / filename))
    
    # Calculate split ratio based on original test/train sizes
    default_ratio = 0.2
    ratio = default_ratio
    
    train_path = original_dir / "train.csv"
    test_path = original_dir / "test.csv"
    
    if train_path.exists() and test_path.exists():
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            train_size = len(train_df)
            test_size = len(test_df)
            
            if train_size > 0:
                calculated_ratio = test_size / train_size
                if 0 < calculated_ratio < 1:
                    ratio = calculated_ratio
                    print(f"Using calculated ratio: {ratio}")
                else:
                    print(f"Calculated ratio {calculated_ratio} is invalid. Using default: {default_ratio}")
            else:
                print(f"Train size is zero. Using default ratio: {default_ratio}")
        except Exception as e:
            print(f"Error calculating ratio: {e}. Using default: {default_ratio}")
    else:
        print(f"Missing train or test files. Using default ratio: {default_ratio}")
    
    # Split original train data into new train and test sets
    print("Splitting original train data...")
    df_original_train = pd.read_csv(train_path)
    
    # Define columns
    label_col = "answer"
    id_col = "id"
    
    # Split the data
    try:
        new_train_df, new_test_df = train_test_split(
            df_original_train, test_size=ratio, random_state=42
        )
    except Exception as e:
        print(f"Error splitting data: {e}. Using default ratio: {default_ratio}")
        new_train_df, new_test_df = train_test_split(
            df_original_train, test_size=default_ratio, random_state=42
        )
    
    # Create test_answer.csv in private directory
    test_answer_df = new_test_df[[id_col, label_col]].copy()
    test_answer_df.to_csv(private / "test_answer.csv", index=False)
    
    # Remove label column from test set
    new_test_df = new_test_df.drop(columns=[label_col])
    
    # Save new train and test files
    new_train_df.to_csv(public / "train.csv", index=False)
    new_test_df.to_csv(public / "test.csv", index=False)
    
    # Create sample_submission.csv
    sample_sub_df = test_answer_df.copy()
    sample_sub_df[label_col] = "A B C"
    sample_sub_df.to_csv(public / "sample_submission.csv", index=False)
    
    # Replicate additional folder structure if needed
    for folder_name in ["train", "test"]:
        source_folder = raw / folder_name
        if source_folder.exists() and source_folder.is_dir():
            target_folder = public / folder_name
            target_folder.mkdir(exist_ok=True)
            
            for root, dirs, files in os.walk(str(source_folder)):
                rel_path = os.path.relpath(root, str(source_folder))
                target_subfolder = target_folder / rel_path
                target_subfolder.mkdir(exist_ok=True, parents=True)
                
                for file in files:
                    src_file = Path(root) / file
                    dst_file = target_subfolder / file
                    shutil.copy2(str(src_file), str(dst_file))
    
    # Validate the split
    assert len(new_train_df) + len(new_test_df) == len(df_original_train), "Split size mismatch"
    
    # Copy additional files from raw to public
    for item in os.listdir(str(raw)):
        item_path = raw / item
        if item != "original" and item_path.is_file():
            if item not in ["train.csv", "test.csv", "sample_submission.csv"]:
                shutil.copy2(str(item_path), str(public))
    
    print("Data preparation complete.")

    # Clean up
    shutil.rmtree(raw)
    # Delete zip file of public/
    for file in os.listdir(public):
        if file.endswith('.zip'):
            os.remove(os.path.join(public, file))
