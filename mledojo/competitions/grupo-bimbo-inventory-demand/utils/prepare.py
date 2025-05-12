import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
from pathlib import Path

def prepare(raw: Path, public: Path, private: Path):
    """
    Prepares data for the Grupo Bimbo Inventory Demand competition by:
    1. Creating new train/test splits from the original training data
    2. Creating sample_submission.csv and test_answer.csv files
    3. Organizing data into appropriate directories
    
    Args:
        raw: Path to the raw data directory
        public: Path to the public data directory
        private: Path to the private data directory
    """
    print("Starting data preparation process...")
    
    # Create output directories if they don't exist
    public.mkdir(exist_ok=True, parents=True)
    private.mkdir(exist_ok=True, parents=True)
    print("Created output directories")
    
    # rm grupo-bimbo-inventory-demand.zip
    os.system("rm grupo-bimbo-inventory-demand.zip")

    # unzip other zip files
    for file in raw.glob("*.zip"):
        os.system(f"unzip {file} -d {raw}")

    # Read the original datasets
    print("Loading original datasets...")
    # Read data in chunks to avoid memory issues
    print("Loading train.csv...")
    train_chunks = pd.read_csv(raw / "train.csv", chunksize=1000000)
    train_df = pd.DataFrame()
    chunk_count = 0
    for chunk in train_chunks:
        train_df = pd.concat([train_df, chunk], ignore_index=True)
        chunk_count += 1
        print(f"Processed {chunk_count} chunks of train.csv ({chunk_count * 1000000} rows)")
    
    print("Loading test.csv...")
    test_chunks = pd.read_csv(raw / "test.csv", chunksize=1000000)
    test_df = pd.DataFrame()
    chunk_count = 0
    for chunk in test_chunks:
        test_df = pd.concat([test_df, chunk], ignore_index=True)
        chunk_count += 1
        print(f"Processed {chunk_count} chunks of test.csv ({chunk_count * 1000000} rows)")
    
    print(f"Original dataset: {len(train_df)} training samples, {len(test_df)} test samples")
    
    # Calculate the split ratio based on original data
    try:
        test_ratio = len(test_df) / (len(train_df) + len(test_df))
        print(f"Calculated test ratio: {test_ratio}")
        if test_ratio > 1 or test_ratio < 0.05:
            print(f"Calculated test ratio {test_ratio} is improper, using default value of 0.2")
            test_ratio = 0.2
    except Exception as e:
        print(f"Error calculating test ratio: {e}, using default value of 0.2")
        test_ratio = 0.2
    
    print(f"Using test ratio: {test_ratio}")
    
    # Split the original training data into new train and test sets
    new_train_df, new_test_df = train_test_split(
        train_df, 
        test_size=test_ratio,
        random_state=42
    )
    
    print(f"Split data: {len(new_train_df)} new training samples, {len(new_test_df)} new test samples")
    
    # Verify the split is correct
    assert len(new_train_df) + len(new_test_df) == len(train_df), "Split sizes don't add up to original size"
    assert abs(len(new_test_df) / len(train_df) - test_ratio) < 0.01, "Test ratio is not as expected"
    
    # Reset index for new_train_df to ensure no overlap with test ids
    new_train_df = new_train_df.reset_index(drop=True)
    
    # Save the new training file
    new_train_df.to_csv(public / "train.csv", index=False)
    print(f"Created new training file")
    
    # Create test.csv (without the target variable)
    # Add an id column to identify each row in the test set
    new_test_df = new_test_df.reset_index(drop=True)
    new_test_df_for_test = new_test_df.copy()
    
    # Create unique IDs that don't overlap with train.csv
    # Start IDs after the last training index
    new_test_df_for_test['id'] = new_test_df_for_test.index + len(new_train_df)
    
    # Remove the target column and other columns not present in the original test set
    columns_to_keep = ['id', 'Semana', 'Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID']
    new_test_df_for_test = new_test_df_for_test[columns_to_keep]
    
    new_test_df_for_test.to_csv(public / "test.csv", index=False)
    print(f"Created new test file")
    
    # Create sample_submission.csv
    sample_submission = pd.DataFrame({
        'id': new_test_df_for_test['id'],
        'Demanda_uni_equil': [0] * len(new_test_df_for_test)
    })
    sample_submission.to_csv(public / "sample_submission.csv", index=False)
    print(f"Created sample_submission.csv")
    
    # Create test_answer.csv with the same IDs as test.csv and sample_submission.csv
    test_answer = pd.DataFrame({
        'id': new_test_df_for_test['id'],
        'Demanda_uni_equil': new_test_df['Demanda_uni_equil'].values
    })
    test_answer.to_csv(private / "test_answer.csv", index=False)
    print(f"Created test_answer.csv")
    
    # Verify test_answer.csv and sample_submission.csv have the same columns
    sample_submission_cols = list(pd.read_csv(public / "sample_submission.csv").columns)
    test_answer_cols = list(pd.read_csv(private / "test_answer.csv").columns)
    assert sample_submission_cols == test_answer_cols, "Column mismatch between test_answer.csv and sample_submission.csv"
    print("Verified test_answer.csv and sample_submission.csv have the same columns")
    
    # Copy additional files to public directory
    additional_files = [
        "cliente_tabla.csv",
        "producto_tabla.csv",
        "town_state.csv",
    ]
    
    for file_name in additional_files:
        file_path = raw / file_name
        if file_path.exists():
            shutil.copy(file_path, public)
            print(f"Copied {file_name}")
    
    print("Data preparation complete!")

    # clean up
    shutil.rmtree(raw)