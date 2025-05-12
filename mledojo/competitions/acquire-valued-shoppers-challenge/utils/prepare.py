import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
from pathlib import Path

def prepare(raw: Path, public: Path, private: Path):
    print("Starting data preparation...")
    
    # Create output directories if they don't exist
    public.mkdir(exist_ok=True, parents=True)
    private.mkdir(exist_ok=True, parents=True)
    
    print("Created output directories")
    
    # Load the data
    print("Loading data files...")

    # extract the gz files
    os.system(f"gunzip {raw / 'trainHistory.csv.gz'}")
    os.system(f"gunzip {raw / 'testHistory.csv.gz'}")
    os.system(f"gunzip {raw / 'offers.csv.gz'}")
    os.system(f"gunzip {raw / 'transactions.csv.gz'}")
    
    # Load main files
    train_history = pd.read_csv(raw / "trainHistory.csv")
    test_history = pd.read_csv(raw / "testHistory.csv")
    offers = pd.read_csv(raw / "offers.csv")
    
    print(f"Loaded trainHistory.csv: {train_history.shape[0]} rows")
    print(f"Loaded testHistory.csv: {test_history.shape[0]} rows")
    print(f"Loaded offers.csv: {offers.shape[0]} rows")
    
    # Copy additional files
    print("Copying additional files to public directory...")
    shutil.copy(raw / "offers.csv", public / "offers.csv")
    
    # Copy transactions.csv - this might be a large file, so print a message
    print("Copying transactions.csv to public directory (this might take some time for large files)...")
    shutil.copy(raw / "transactions.csv", public / "transactions.csv")
    
    # Calculate split ratio based on original test/train sizes
    try:
        original_train_size = train_history.shape[0]
        original_test_size = test_history.shape[0]
        test_ratio = original_test_size / (original_train_size + original_test_size)
        
        print(f"Original train size: {original_train_size}")
        print(f"Original test size: {original_test_size}")
        print(f"Calculated test ratio: {test_ratio:.4f}")
        
        # Validate the ratio
        if test_ratio <= 0.05 or test_ratio >= 1:
            print("Warning: Calculated ratio is improper. Using default ratio of 0.2")
            test_ratio = 0.2
    except Exception as e:
        print(f"Error calculating split ratio: {e}")
        print("Using default ratio of 0.2")
        test_ratio = 0.2
    
    # Perform train-test split on trainHistory.csv
    print(f"Performing train-test split with test_ratio={test_ratio:.4f}...")
    train_set, test_set = train_test_split(train_history, test_size=test_ratio, random_state=42)
    
    print(f"New train set size: {train_set.shape[0]}")
    print(f"New test set size: {test_set.shape[0]}")
    
    # Validate the split
    assert len(train_set) + len(test_set) == len(train_history), "Split validation failed: Total rows don't match"
    assert abs(len(test_set) / len(train_history) - test_ratio) < 0.01, "Split validation failed: Ratio doesn't match"
    
    print("Train-test split validation passed")
    
    # Save the new train set
    train_set.to_csv(public / "trainHistory.csv", index=False)
    print("Saved new trainHistory.csv")
    
    # Create new test set (without repeater and repeattrips columns)
    new_test = test_set.drop(['repeater', 'repeattrips'], axis=1)
    new_test.to_csv(public / "testHistory.csv", index=False)
    print("Saved new testHistory.csv")
    
    # Create test_answer.csv
    test_answers = test_set[['id', 'repeater']].copy()
    # Convert boolean 't'/'f' to numeric 1/0 probabilities for the ROC curve evaluation
    test_answers['repeatProbability'] = test_answers['repeater'].map({'t': 1.0, 'f': 0.0})
    test_answers = test_answers[['id', 'repeatProbability']]
    test_answers.to_csv(private / "test_answer.csv", index=False)
    print("Saved test_answer.csv")
    
    # Create sample_submission.csv
    sample_submission = pd.DataFrame({
        'id': new_test['id'],
        'repeatProbability': np.zeros(len(new_test))
    })
    sample_submission.to_csv(public / "sample_submission.csv", index=False)
    print("Saved sample_submission.csv")
    
    # Validate that test_answer.csv and sample_submission.csv have the same columns
    test_answer = pd.read_csv(private / "test_answer.csv")
    sample_submission = pd.read_csv(public / "sample_submission.csv")
    
    assert list(test_answer.columns) == list(sample_submission.columns), "Validation failed: Columns in test_answer.csv and sample_submission.csv don't match"
    print("Column validation passed: test_answer.csv and sample_submission.csv have matching columns")
    
    print("Data preparation completed successfully!")
    
    # clean up
    shutil.rmtree(raw)
    