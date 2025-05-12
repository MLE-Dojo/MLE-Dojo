import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
from pathlib import Path

def prepare(raw: Path, public: Path, private: Path):
    """
    Prepare data for the Allstate Purchase Prediction Challenge by splitting the original
    training data into new train and test sets, and creating necessary submission files.
    
    Args:
        raw: Path to the raw data directory
        public: Path to the public output directory
        private: Path to the private output directory
    """
    print("Starting data preparation...")
    
    # Create output directories if they don't exist
    os.makedirs(public, exist_ok=True)
    os.makedirs(private, exist_ok=True)
    
    # unzip the data
    shutil.unpack_archive(raw / 'train.csv.zip', raw )
    shutil.unpack_archive(raw / 'test_v2.csv.zip', raw)

    print(f"Created output directories: {public} and {private}")
    
    # Load datasets
    print("Loading datasets...")
    train_df = pd.read_csv(raw / 'train.csv')
    try:
        test_v2_df = pd.read_csv(raw / 'test_v2.csv')
        original_test_size = test_v2_df['customer_ID'].nunique()
        print(f"Original test set has {original_test_size} unique customers")
    except:
        print("Could not find or load test_v2.csv, will use default split ratio")
        original_test_size = None
    
    # Get unique customers in training data
    unique_customers = train_df['customer_ID'].unique()
    total_customers = len(unique_customers)
    print(f"Original training set has {total_customers} unique customers")
    
    # Calculate split ratio based on original data sizes
    if original_test_size is not None:
        split_ratio = original_test_size / (original_test_size + total_customers)
        
        # Validate split ratio
        if split_ratio <= 0.05 or split_ratio >= 1:
            print(f"Invalid split ratio calculated: {split_ratio}, using default of 0.2")
            split_ratio = 0.2
        else:
            print(f"Using calculated split ratio: {split_ratio}")
    else:
        split_ratio = 0.2
        print(f"Using default split ratio: {split_ratio}")
    
    # Split customers into train and test sets
    print("Splitting data...")
    train_customers, test_customers = train_test_split(unique_customers, test_size=split_ratio, random_state=42)
    
    print(f"New train set will have {len(train_customers)} customers")
    print(f"New test set will have {len(test_customers)} customers")
    
    # Get all transactions for each customer
    train_data = train_df[train_df['customer_ID'].isin(train_customers)]
    test_data_all = train_df[train_df['customer_ID'].isin(test_customers)]
    
    # For test data, exclude purchase points (record_type=1) to simulate real test environment
    test_data = test_data_all[test_data_all['record_type'] == 0].copy()
    
    # Create test_answer.csv containing the purchased options
    test_answers = test_data_all[test_data_all['record_type'] == 1].copy()
    
    if len(test_answers) == 0:
        print("No purchase records found in test data! Extracting last record for each customer instead.")
        # If no purchase records, get the last shopping point for each customer
        test_answers = test_data_all.groupby('customer_ID').apply(lambda x: x.iloc[-1]).reset_index(drop=True)
    
    # Create sample_submission.csv
    print("Creating submission files...")
    
    # Create test_answer.csv with purchased plan
    test_answers['plan'] = ''
    for col in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
        test_answers['plan'] = test_answers['plan'] + test_answers[col].astype(str)
    
    test_answers = test_answers[['customer_ID', 'plan']]
    
    # Create sample_submission.csv with the same customer IDs as test_answer.csv
    sample_submission = pd.DataFrame({'customer_ID': test_answers['customer_ID'].unique()})
    sample_submission['plan'] = '1111111'  # Default value as in original sample submission
    
    # Save files
    print("Saving files...")
    train_data.to_csv(public / 'train.csv', index=False)
    test_data.to_csv(public / 'test.csv', index=False)
    sample_submission.to_csv(public / 'sample_submission.csv', index=False)
    test_answers.to_csv(private / 'test_answer.csv', index=False)
    
    
    # Validation checks
    print("Running validation checks...")
    # Check if train-test split is proper
    assert len(train_customers) + len(test_customers) == total_customers, "Train-test split resulted in missing customers"
    assert len(set(train_customers).intersection(set(test_customers))) == 0, "Train and test sets have overlapping customers"
    
    # Check if test_answer.csv and sample_submission.csv have the same columns
    test_answer_df = pd.read_csv(private / 'test_answer.csv')
    sample_submission_df = pd.read_csv(public / 'sample_submission.csv')
    
    assert set(test_answer_df.columns) == set(sample_submission_df.columns), "test_answer.csv and sample_submission.csv have different columns"
    
    # Check if test_answer.csv and sample_submission.csv have the same customer IDs
    assert set(test_answer_df['customer_ID']) == set(sample_submission_df['customer_ID']), "test_answer.csv and sample_submission.csv have different customer IDs"
    
    print("All checks passed!")
    print(f"Files saved to public directory: {public}")
    print(f"Test answers saved to private directory: {private}")
    print("Data preparation completed successfully!")

    # clean up
    shutil.rmtree(raw)