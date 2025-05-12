import os
import shutil
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path

def prepare(raw: Path, public: Path, private: Path):
    print("[INFO] Starting data preparation...")

    # Directories
    data_raw_dir = raw
    data_public_dir = public
    data_private_dir = private
    original_dir = os.path.join(data_raw_dir, 'original')

    # Ensure output directories exist
    if not os.path.exists(data_public_dir):
        print(f"[INFO] Creating directory: {data_public_dir}")
        os.makedirs(data_public_dir, exist_ok=True)
    if not os.path.exists(data_private_dir):
        print(f"[INFO] Creating directory: {data_private_dir}")
        os.makedirs(data_private_dir, exist_ok=True)

    # 1. Check if 'original' folder exists, if not create and move train/test files there
    if not os.path.exists(original_dir):
        print(f"[INFO] Creating 'original' folder at {original_dir} and moving train/test files.")
        os.makedirs(original_dir, exist_ok=True)
        # Move known files if they exist
        for filename in ['train.csv', 'test.csv', 'sample_submission.csv']:
            src = os.path.join(data_raw_dir, filename)
            if os.path.exists(src):
                dst = os.path.join(original_dir, filename)
                shutil.move(src, dst)
    else:
        print(f"[INFO] 'original' folder already exists. Using existing files.")

    # Paths to original train and test
    orig_train_path = os.path.join(original_dir, 'train.csv')
    orig_test_path = os.path.join(original_dir, 'test.csv')

    # 2. Calculate split ratio based on original train/test sizes
    default_ratio = 0.2
    ratio = default_ratio
    ratio_error = False

    try:
        if os.path.exists(orig_train_path) and os.path.exists(orig_test_path):
            df_train_orig = pd.read_csv(orig_train_path)
            df_test_orig = pd.read_csv(orig_test_path)

            train_size = len(df_train_orig)
            test_size = len(df_test_orig)
            print(f"[INFO] Original train size: {train_size}, test size: {test_size}")

            if train_size > 0:
                calculated_ratio = test_size / train_size
                # Check if ratio is valid
                if calculated_ratio <= 0 or calculated_ratio > 1:
                    ratio_error = True
                    print(f"[ERROR] Invalid ratio calculated: {calculated_ratio}. Using default ratio {default_ratio}.")
                else:
                    ratio = calculated_ratio
            else:
                ratio_error = True
                print(f"[ERROR] Original train file has zero rows. Using default ratio {default_ratio}.")
        else:
            ratio_error = True
            print(f"[ERROR] Original train/test files not found. Using default ratio {default_ratio}.")
    except Exception as e:
        ratio_error = True
        print(f"[ERROR] Exception occurred while calculating ratio: {e}. Using default ratio {default_ratio}.")

    print(f"[INFO] Final split ratio used: {ratio}")

    # 3. Using this ratio, create new train and test files from the original train file
    # If we haven't loaded df_train_orig above, load it now
    if not 'df_train_orig' in locals():
        if os.path.exists(orig_train_path):
            df_train_orig = pd.read_csv(orig_train_path)
        else:
            print("[ERROR] Cannot find original train file. Exiting.")
            return

    # Shuffle / split
    df_train_orig = df_train_orig.sample(frac=1.0, random_state=42).reset_index(drop=True)  # shuffle
    split_index = int(len(df_train_orig)*(1-ratio))
    new_train_df = df_train_orig.iloc[:split_index].copy()
    new_test_df = df_train_orig.iloc[split_index:].copy()

    # 4. Identify the label/target and Id columns based on the data fields
    #    We assume 'target' is the label and 'Id' is the unique ID
    target_col = 'target'
    id_col = 'Id'

    # 5. The test files should not contain the label/target column
    new_test_no_target = new_test_df.drop(columns=[target_col])

    # 6. Save the label/target column along with the id of the test data to ../data/private/test_answer.csv
    test_answer = new_test_df[[id_col, target_col]].copy()
    test_answer.rename(columns={target_col: 'Predicted'}, inplace=True)
    test_answer_path = os.path.join(data_private_dir, 'test_answer.csv')
    test_answer.to_csv(test_answer_path, index=False)
    print(f"[INFO] Saved test_answer.csv to {test_answer_path}")



    # 7. Create sample_submission.csv matching test_answer.csv columns but with different answer values
    #    The official submission format is typically: Id,Predicted
    sample_sub = test_answer[[id_col]].copy()
    sample_sub['Predicted'] = 0  # or any placeholder
    sample_sub_path = os.path.join(data_public_dir, 'sample_submission.csv')
    sample_sub.to_csv(sample_sub_path, index=False)
    print(f"[INFO] Saved sample_submission.csv to {sample_sub_path}")

    # 8. Optionally Zip the newly generated train and test files if the size is greater than 2Mb
    #    We first save them normally, then check size and zip if needed
    new_train_path = os.path.join(data_public_dir, 'train.csv')
    new_test_path = os.path.join(data_public_dir, 'test.csv')

    new_train_df.to_csv(new_train_path, index=False)
    new_test_no_target.to_csv(new_test_path, index=False)


    # 9. Copy any other additional files from data_raw other than the old test/train files to ../data/public
    #    We'll skip any known files (train.csv, test.csv, sample_submission.csv) even if they're in original dir.
    known_files = {'train.csv', 'test.csv', 'sample_submission.csv'}
    for item in os.listdir(data_raw_dir):
        # skip 'original' folder itself
        if item == 'original':
            continue
        src_item = os.path.join(data_raw_dir, item)
        dst_item = os.path.join(data_public_dir, item)
        if os.path.isfile(src_item):
            if item not in known_files:
                print(f"[INFO] Copying additional file '{item}' to public directory.")
                shutil.copy2(src_item, dst_item)
        elif os.path.isdir(src_item):
            # if there is any directory other than 'original', copy the entire folder
            if item != 'original':
                print(f"[INFO] Copying directory '{item}' to public directory.")
                if os.path.exists(dst_item):
                    shutil.rmtree(dst_item)
                shutil.copytree(src_item, dst_item)

    # 10. Validate with assertions for checking if the test train split is proper
    #     Check that train + test = original train size
    assert len(new_train_df) + len(new_test_df) == len(df_train_orig), \
        "[ASSERT ERROR] The new train + new test sizes do not match the original train size."

    # Check that target is removed from the new test data
    assert target_col not in new_test_no_target.columns, \
        "[ASSERT ERROR] Target column still present in the new test file."

    print("[INFO] Data preparation completed successfully.")

    # Clean up
    for file in os.listdir(data_public_dir):
        if file.endswith('.zip'):
            os.remove(os.path.join(data_public_dir, file))
    shutil.rmtree(data_raw_dir)