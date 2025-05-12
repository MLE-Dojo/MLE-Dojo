import os
import shutil
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def prepare(raw: Path, public: Path, private: Path):
    """
    Prepare data by splitting raw data into public and private datasets.
    
    Args:
        raw: Path to raw data directory
        public: Path to public data directory
        private: Path to private data directory
    """
    print("Starting data preparation...")
    
    # Create output directories if they don't exist
    os.makedirs(public, exist_ok=True)
    os.makedirs(private, exist_ok=True)
    print(f"Ensured directories: {public} and {private}")
    
    # Load train and test data from raw data directory
    train_file = raw / "train.csv"
    test_file = raw / "test.csv"
    print("Loading train.csv and test.csv...")
    df_train = pd.read_csv(train_file)
    df_test_orig = pd.read_csv(test_file)
    print(f"Loaded train.csv with {len(df_train)} rows and test.csv with {len(df_test_orig)} rows.")
    
    # Calculate split ratio based on original test/train sizes
    try:
        ratio = len(df_test_orig) / len(df_train)
        if ratio > 1 or ratio < 0.05:
            print(f"Calculated ratio {ratio:.4f} is out of acceptable bounds. Using default ratio 0.2")
            ratio = 0.2
        else:
            print(f"Calculated split ratio: {ratio:.4f}")
    except Exception as e:
        print("Error in calculating ratio:", e)
        ratio = 0.2
        print("Using default ratio: 0.2")
    
    train_size = 1 - ratio
    print(f"New train size will be approximately {int(len(df_train) * train_size)} rows and new test size {int(len(df_train) * ratio)} rows.")
    
    # Split the training data into new_train and new_test splits
    new_train, new_test = train_test_split(df_train, test_size=ratio, random_state=42)
    print("Data split into new_train and new_test.")
    
    # Validations for proper split
    assert len(new_train) + len(new_test) == len(df_train), "Train-test split sizes do not sum up to original size."
    if 'id' in df_train.columns:
        overlap = set(new_train['id']).intersection(set(new_test['id']))
        assert len(overlap) == 0, "Overlap found between new train and test splits."
    print("Train-test split validation passed.")
    
    # Create test_answer.csv (ground truth) in the private folder from new_test (columns: id, trade_price)
    test_answer = new_test[['id', 'trade_price', 'weight']].copy()
    test_answer_file = private / "test_answer.csv"
    test_answer.to_csv(test_answer_file, index=False)
    print(f"Saved test_answer.csv to {test_answer_file}")
    
    # Create sample_submission.csv in the public folder using new_test ids and dummy predictions
    sample_submission = new_test[['id']].copy()
    sample_submission['trade_price'] = 0  # dummy predictions
    sample_submission_file = public / "sample_submission.csv"
    sample_submission.to_csv(sample_submission_file, index=False)
    print(f"Saved sample_submission.csv to {sample_submission_file}")
    
    
    # Save the new train and test (public) splits to the public folder
    new_train_file = public / "train.csv"
    new_test_public_file = public / "test.csv"
    new_train.to_csv(new_train_file, index=False)
    new_test.to_csv(new_test_public_file, index=False)
    print(f"Saved new train.csv to {new_train_file} and new test.csv to {new_test_public_file}")
    
    print("Data preparation completed successfully.")

    # clean up
    shutil.rmtree(raw)
