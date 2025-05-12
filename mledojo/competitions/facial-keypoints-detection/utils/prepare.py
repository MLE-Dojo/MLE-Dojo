import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def prepare(raw: Path, public: Path, private: Path):
    """
    Splits the data in raw into public and private datasets with appropriate test/train splits.
    
    Args:
        raw: Path to the raw data directory
        public: Path to the public output directory
        private: Path to the private output directory
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # recursively unzip all zip files in raw
    os.system(f"rm facial-keypoints-detection.zip ")
    os.system(f"unzip test.zip")
    os.system(f"unzip training.zip")
    os.system(f"rm test.zip training.zip")

    # Create output directories
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    
    # -------------------
    # 1. Read and split training.csv
    # -------------------
    training_csv_path = raw / "training.csv"
    df_train = pd.read_csv(training_csv_path)
    
    # Shuffle rows with a fixed random seed and reset index
    df_train_shuffled = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
    total_rows = len(df_train_shuffled)
    assert total_rows == 7049, f"Expected 7049 rows in training.csv; got {total_rows}"
    
    # Split indices: new training set with 5266 rows, new test set with 1783 rows
    n_train_public = 5266
    n_test = total_rows - n_train_public  # expected 1783
    assert n_train_public + n_test == total_rows
    
    df_public_train = df_train_shuffled.iloc[:n_train_public].copy()    # Features + answers kept
    df_test_split = df_train_shuffled.iloc[n_train_public:].copy()      # Will be divided into public test features and private test answers
    
    # Assert that all original columns are preserved in the new training set:
    assert list(df_public_train.columns) == list(df_train.columns), "Column mismatch in new training set."
    
    # Save new training set to public/training.csv
    public_train_csv = public / "training.csv"
    df_public_train.to_csv(public_train_csv, index=False)
    
    # -------------------
    # 2. Process the new test set (1783 rows)
    # -------------------
    IMAGE_COL = "Image"
    if IMAGE_COL not in df_train.columns:
        raise ValueError(f"Expected image data column '{IMAGE_COL}' not found in training.csv.")
    
    # Create an "ImageId" column for test_split data (using 1-indexing)
    df_test_split = df_test_split.copy()  # Ensure we are not modifying the original dataframe
    df_test_split.insert(0, "ImageId", range(1, len(df_test_split) + 1))
    
    # Build public test features file: keep only 'ImageId' and the image column.
    df_public_test = df_test_split[["ImageId", IMAGE_COL]].copy()
    public_test_csv = public / "test.csv"
    df_public_test.to_csv(public_test_csv, index=False)
    
    # -------------------
    # 2.b Create private test_answer.csv in long (submission) format.
    # -------------------
    keypoint_columns = [col for col in df_train.columns if col != IMAGE_COL]
    expected_num_keypoints = 30
    if len(keypoint_columns) != expected_num_keypoints:
        raise ValueError(f"Expected {expected_num_keypoints} keypoint columns, but found {len(keypoint_columns)}.")
    
    # Process each row in the test split and build the long format.
    private_rows = []
    row_id = 1
    tqdm_desc = "Processing test set for private answers"
    for _, row in tqdm(df_test_split.iterrows(), total=len(df_test_split), desc=tqdm_desc):
        img_id = row["ImageId"]
        for feature in keypoint_columns:
            loc = row[feature]
            private_rows.append((row_id, img_id, feature, loc))
            row_id += 1
    
    # Create DataFrame for private test answers
    df_private_test_answer = pd.DataFrame(private_rows, columns=["RowId", "ImageId", "FeatureName", "Location"])
    
    # Assert the row count is exactly 1783 * 30
    expected_answer_rows = n_test * expected_num_keypoints
    assert len(df_private_test_answer) == expected_answer_rows, \
        f"Expected {expected_answer_rows} rows in test_answer.csv, but got {len(df_private_test_answer)}."
    
    # Clean up missing values in test_answer.csv
    missing_location_mask = df_private_test_answer["Location"].isna()
    rows_with_missing_values = df_private_test_answer[missing_location_mask]
    
    if not rows_with_missing_values.empty:
        print(f"Found {len(rows_with_missing_values)} rows with missing Location values in test_answer.csv")
        
        # Get the RowId values of rows with missing Location
        missing_row_ids = rows_with_missing_values["RowId"].tolist()
        
        # Remove rows with missing Location values from test_answer.csv
        df_private_test_answer = df_private_test_answer[~missing_location_mask]
        
        # Update the expected_answer_rows count
        expected_answer_rows = len(df_private_test_answer)
        print(f"Updated expected_answer_rows to {expected_answer_rows}")
    
    # Save private test answer file
    private_test_answer_csv = private / "test_answer.csv"
    df_private_test_answer.to_csv(private_test_answer_csv, index=False)
    
    # -------------------
    # 3. Build sample_submission.csv in public with dummy answers.
    # -------------------
    df_sample_submission = df_private_test_answer.copy()
    # Fill the "Location" column with dummy (zero) values of numeric type.
    df_sample_submission["Location"] = 0
    
    # Verify column order and names: "RowId, ImageId, FeatureName, Location"
    expected_columns = ["RowId", "ImageId", "FeatureName", "Location"]
    assert list(df_sample_submission.columns) == expected_columns, "sample_submission.csv column mismatch."
    
    public_sample_submission_csv = public / "sample_submission.csv"
    df_sample_submission.to_csv(public_sample_submission_csv, index=False)
    
    # -------------------
    # 4. Copy other files from the original data folder
    # -------------------
    # Copy IdLookupTable.csv into public (unchanged)
    src_id_lookup = raw / "IdLookupTable.csv"
    dst_id_lookup = public / "IdLookupTable.csv"
    shutil.copy(src_id_lookup, dst_id_lookup)
    
    # -------------------
    # Final Assertions & Checks
    # -------------------
    assert pd.read_csv(public_train_csv).shape[0] == n_train_public, "Mismatch in public/training.csv row count."
    assert pd.read_csv(public_test_csv).shape[0] == n_test, "Mismatch in public/test.csv row count."
    assert pd.read_csv(public_sample_submission_csv).shape[0] == expected_answer_rows, \
           "Mismatch in public/sample_submission.csv row count."
    assert (public_sample_submission_csv.exists() and dst_id_lookup.exists()), "Expected public files not found."
    
    # Check private/test_answer.csv row count
    assert pd.read_csv(private_test_answer_csv).shape[0] == expected_answer_rows, \
           "Mismatch in private/test_answer.csv row count."
    
    print("Data preparation completed successfully.")

    # Clean up
    shutil.rmtree(raw)