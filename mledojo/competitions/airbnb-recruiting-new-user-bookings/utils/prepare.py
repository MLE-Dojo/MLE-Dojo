import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
import csv
from pathlib import Path

def prepare(raw: Path, public: Path, private: Path):
    """
    Prepare data for the Airbnb New User Bookings competition by creating a new train-test split
    from the existing training data. Also creates test_answer.csv and sample_submission.csv files.
    
    Args:
        raw: Path to the raw data directory
        public: Path to the public output directory
        private: Path to the private output directory
    """
    print("Starting data preparation...")
    
    # Create output directories if they don't exist
    for directory in [public, private]:
        if not directory.exists():
            directory.mkdir(parents=True)
    
    print(f"Created output directories: {public} and {private}")

    # unzip all zip files in the raw directory to the raw directory
    # except for airbnb-recruiting-new-user-bookings.zip
    for file in raw.glob("*.zip"):
        if file.name != "airbnb-recruiting-new-user-bookings.zip":
            os.system(f"unzip {file} -d {raw}")

    
    # Load the original datasets
    print("Loading datasets...")
    
    try:
        train_users = pd.read_csv(raw / "train_users_2.csv")
        test_users = pd.read_csv(raw / "test_users.csv")
        
        # Load additional files to copy
        sessions = None
        countries = None
        age_gender_bkts = None
        
        if (raw / "sessions.csv").exists():
            print("Loading sessions.csv...")
            sessions = pd.read_csv(raw / "sessions.csv")
        
        if (raw / "countries.csv").exists():
            print("Loading countries.csv...")
            countries = pd.read_csv(raw / "countries.csv")
        
        if (raw / "age_gender_bkts.csv").exists():
            print("Loading age_gender_bkts.csv...")
            age_gender_bkts = pd.read_csv(raw / "age_gender_bkts.csv")
            
        print("All datasets loaded successfully")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return
    
    # Calculate train/test split ratio based on original sizes
    try:
        original_train_size = len(train_users)
        original_test_size = len(test_users)
        original_ratio = original_test_size / original_train_size
        
        print(f"Original train size: {original_train_size}")
        print(f"Original test size: {original_test_size}")
        print(f"Original test/train ratio: {original_ratio:.4f}")
        
        # Validate ratio
        if original_ratio <= 0.05 or original_ratio >= 1.0:
            print(f"Calculated ratio ({original_ratio:.4f}) is outside acceptable range [0.05, 1.0]")
            print("Using default ratio of 0.2")
            test_size = 0.2
        else:
            test_size = original_ratio
            print(f"Using calculated test_size: {test_size:.4f}")
    
    except Exception as e:
        print(f"Error calculating split ratio: {e}")
        print("Using default ratio of 0.2")
        test_size = 0.2
    
    # Create new train/test split
    print("Creating new train/test split...")
    new_train, new_test = train_test_split(
        train_users, test_size=test_size, random_state=42, stratify=train_users['country_destination']
    )
    
    print(f"New train size: {len(new_train)}")
    print(f"New test size: {len(new_test)}")
    
    # Create test_answer.csv
    print("Creating test_answer.csv...")
    test_answers = new_test[['id', 'country_destination']].copy()

    # change the column name to country
    test_answers.rename(columns={'country_destination': 'country'}, inplace=True)
    
    
    # Create sample_submission.csv
    print("Creating sample_submission.csv...")
    
    # We'll create a submission with just 'NDF' for all users as a baseline
    sample_submission_rows = []
    for user_id in new_test['id']:
        sample_submission_rows.append({'id': user_id, 'country': 'NDF'})
    
    sample_submission = pd.DataFrame(sample_submission_rows)
    
    # Validate data split and file structures
    print("Validating data...")
    
    # Check that train and test sets are properly split
    assert len(new_train) + len(new_test) == len(train_users), "Train and test split size doesn't match original data"
    assert abs((len(new_test) / len(train_users)) - test_size) < 0.01, "Test split ratio doesn't match expected value"
    
    # Check that test_answer and sample_submission have the same columns
    assert set(test_answers.columns) == set(['id', 'country']), "Test answer columns don't match expected format"
    assert set(sample_submission.columns) == set(['id', 'country']), "Sample submission columns don't match expected format"
    
    # Save files
    print("Saving files...")
    
    # Save main files
    new_train.to_csv(public / "train_users.csv", index=False)
    print(f"Saved: {public / 'train_users.csv'}")
    
    # For the test file, we need to remove the target column like in the original test file
    new_test_without_target = new_test.drop(columns=['country_destination'])
    new_test_without_target.to_csv(public / "test_users.csv", index=False)
    print(f"Saved: {public / 'test_users.csv'}")
    
    test_answers.to_csv(private / "test_answer.csv", index=False)
    print(f"Saved: {private / 'test_answer.csv'}")
    
    sample_submission.to_csv(public / "sample_submission.csv", index=False)
    print(f"Saved: {public / 'sample_submission.csv'}")
    
    # Copy additional files
    if sessions is not None:
        sessions.to_csv(public / "sessions.csv", index=False)
        print(f"Saved: {public / 'sessions.csv'}")
    
    if countries is not None:
        countries.to_csv(public / "countries.csv", index=False)
        print(f"Saved: {public / 'countries.csv'}")
    
    if age_gender_bkts is not None:
        age_gender_bkts.to_csv(public / "age_gender_bkts.csv", index=False)
        print(f"Saved: {public / 'age_gender_bkts.csv'}")
    
    print("Data preparation completed successfully!")

    # clean up
    shutil.rmtree(raw)
