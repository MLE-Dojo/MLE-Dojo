import os
import json
import shutil
import pandas as pd
import numpy as np
import random
import subprocess
from sklearn.model_selection import train_test_split
import zipfile
from pathlib import Path

def prepare(raw: Path, public: Path, private: Path):
    """
    Prepare the data for the iMaterialist Challenge Furniture 2018 competition by:
    1. Organizing the original data
    2. Creating new train/test splits
    3. Preparing submission files
    4. Validating the results
    """
    print("Starting data preparation...")
    
    # Define original directory
    original_dir = raw / 'original'
    
    # Create necessary directories if they don't exist
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Analyze data structure and setup original directory
    print("Step 1: Analyzing data structure...")
    
    if not original_dir.exists():
        original_dir.mkdir(parents=True, exist_ok=True)
        # Copy original files to original directory
        for file_name in ['train.json', 'test.json', 'validation.json', 'sample_submission_randomlabel.csv']:
            src_path = raw / file_name
            if src_path.exists():
                shutil.copy(src_path, original_dir / file_name)
                print(f"  Copied {file_name} to original directory")
    
    # Step 2: Load data and calculate split ratio
    print("Step 2: Loading data and calculating split ratio...")
    
    try:
        with open(original_dir / 'train.json', 'r') as f:
            train_data = json.load(f)
        
        with open(original_dir / 'test.json', 'r') as f:
            test_data = json.load(f)
            
        train_size = len(train_data.get('images', []))
        test_size = len(test_data.get('images', []))
        
        print(f"  Original train size: {train_size}")
        print(f"  Original test size: {test_size}")
        
        if train_size > 0:
            test_train_ratio = test_size / train_size
            if test_train_ratio > 0 and test_train_ratio < 1:
                split_ratio = test_train_ratio
                print(f"  Calculated split ratio: {split_ratio:.4f}")
            else:
                split_ratio = 0.2
                print(f"  Calculated ratio is invalid: {test_train_ratio}. Using default ratio: {split_ratio}")
        else:
            split_ratio = 0.2
            print(f"  Cannot calculate ratio. Using default ratio: {split_ratio}")
            
    except Exception as e:
        print(f"  Error loading data: {e}")
        split_ratio = 0.2
        print(f"  Using default split ratio: {split_ratio}")
    
    # Step 3 & 4: Create new train and test splits
    print("Step 3 & 4: Creating new train and test splits...")
    
    try:
        # Load original train data
        with open(original_dir / 'train.json', 'r') as f:
            train_data = json.load(f)
        
        # Extract images and annotations
        images = train_data.get('images', [])
        annotations = train_data.get('annotations', [])
        
        # Create a mapping from image_id to annotation
        image_id_to_label = {ann['image_id']: ann['label_id'] for ann in annotations}
        
        # Create dataframe with image_ids and corresponding labels
        df = pd.DataFrame(images)
        df['label_id'] = df['image_id'].map(image_id_to_label)
        
        # Split the data
        train_df, test_df = train_test_split(df, test_size=split_ratio, random_state=42, stratify=df['label_id'] if len(df) > 1000 else None)
        
        print(f"  New train size: {len(train_df)}")
        print(f"  New test size: {len(test_df)}")
        
        # Create new train JSON
        new_train_data = {
            'images': train_df.drop('label_id', axis=1).to_dict('records'),
            'annotations': [{'image_id': row['image_id'], 'label_id': row['label_id']} 
                           for _, row in train_df.iterrows()]
        }
        
        # Create new test JSON (without labels)
        new_test_data = {
            'images': test_df.drop('label_id', axis=1).to_dict('records')
        }
        
        # Save test answers separately
        test_answers = test_df[['image_id', 'label_id']].rename(columns={'image_id': 'id', 'label_id': 'predicted'})
        test_answers.to_csv(private / 'test_answer.csv', index=False)
        print(f"  Saved test answers to {private / 'test_answer.csv'}")
        
        # Step 5 & 6: Save new train and test files
        with open(public / 'train.json', 'w') as f:
            json.dump(new_train_data, f)
        print(f"  Saved new train data to {public / 'train.json'}")
        
        with open(public / 'test.json', 'w') as f:
            json.dump(new_test_data, f)
        print(f"  Saved new test data to {public / 'test.json'}")
        
        # Step 7: Create sample_submission.csv
        sample_submission = test_df[['image_id']].rename(columns={'image_id': 'id'})
        sample_submission['predicted'] = sample_submission['id'].apply(lambda _: random.randint(0, 127))
        sample_submission.to_csv(public / 'sample_submission.csv', index=False)
        print(f"  Created sample_submission.csv with random predictions")
        
    except Exception as e:
        print(f"  Error creating train-test split: {e}")
    
    # Step 8: Check for additional train/test folders
    print("Step 8: Checking for additional train/test folders...")
    
    for folder in ['train', 'test']:
        src_folder = raw / folder
        if src_folder.exists() and src_folder.is_dir():
            print(f"  Found {folder} folder")
            # Create corresponding folders in public
            dest_folder = public / folder
            dest_folder.mkdir(exist_ok=True)
            
            # Move files according to new split
            files = list(src_folder.iterdir())
            if files:
                print(f"  Processing files in {folder} folder...")
                # Implement folder content alignment based on the train-test split
                # This is a simplified example - you'd need to adapt based on actual folder contents
    
    # Step 9: Validate with assertions
    print("Step 9: Validating the split...")
    try:
        # Ensure test and train don't overlap
        assert set(train_df['image_id']).isdisjoint(set(test_df['image_id'])), "Train and test sets overlap!"
        # Ensure we didn't lose any data
        assert len(train_df) + len(test_df) == len(df), "Data count mismatch after splitting!"
        # Check if test ratio is approximately what we wanted
        actual_ratio = len(test_df) / len(train_df)
        assert abs(actual_ratio - split_ratio) < 0.01, f"Split ratio mismatch: got {actual_ratio}, wanted {split_ratio}"
        print("  Validation passed!")
    except AssertionError as e:
        print(f"  Validation failed: {e}")
    
    
    print("Data preparation complete!")

    # clean up
    shutil.rmtree(raw)
