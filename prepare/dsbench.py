# for DS Bench competitions
import wget
import os
import shutil
import pandas as pd
import numpy as np
import sys
from typing import List
from argparse import ArgumentParser
from pathlib import Path # Import pathlib

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--raw-dir", type=str, required=True)
    parser.add_argument("--prepared-dir", type=str, required=True)
    return parser.parse_args()

def download_and_extract_data(base_dir: str) -> None:
    """Download and extract the DSBench dataset"""
    # Download the zip file
    data_url = "https://huggingface.co/datasets/liqiang888/DSBench/resolve/main/data_modeling/data.zip"
    zip_path = os.path.join(base_dir, "data.zip")
    mac_os_x_path = os.path.join(base_dir, "__MACOSX")
    print(f"Downloading data from {data_url}...")
    wget.download(data_url, zip_path)
    
    # Unzip the file
    print("\nExtracting data...")
    os.system(f"unzip {zip_path} -d {base_dir}")
    os.system(f"rm -r {mac_os_x_path}")
    os.system(f"rm -r {zip_path}")
    print("Data extraction complete.")

def get_folder_lists(data_dir: str) -> tuple:
    """Get lists of folders and files in the data directory"""
    data_resplit_dir = os.path.join(data_dir, "data_resplit")
    answers_dir = os.path.join(data_dir, "answers")
    task_dir = os.path.join(data_dir, "task")
    
    data_resplit_folders = [f for f in os.listdir(data_resplit_dir) if os.path.isdir(os.path.join(data_resplit_dir, f))]
    answers_folders = [f for f in os.listdir(answers_dir) if os.path.isdir(os.path.join(answers_dir, f))]
    task_files = [f.split('.')[0] for f in os.listdir(task_dir) if f.endswith('.txt')]
    
    return data_resplit_folders, answers_folders, task_files, data_resplit_dir, answers_dir, task_dir

def fix_submission_files(folders: List[str], data_resplit_dir: str) -> None:
    """Fix and standardize submission file names"""
    # Fix specific file naming issue
    os.rename(
        os.path.join(data_resplit_dir, "tabular-playground-series-sep-2021", "sample_solution.csv"),
        os.path.join(data_resplit_dir, "tabular-playground-series-sep-2021", "sample_submission.csv")
    )
    
    # Check and rename submission files in all folders
    for folder in folders:
        folder_path = os.path.join(data_resplit_dir, folder)
        submission_files = [f for f in os.listdir(folder_path) if 'submission' in f.lower() and f.endswith('.csv')]
        
        if not submission_files:
            print(f"Error: Folder {folder} does not contain any submission CSV file")
            sys.exit(1)
        
        # Rename submission file to sample_submission.csv if needed
        for submission_file in submission_files:
            if submission_file != 'sample_submission.csv':
                src_path = os.path.join(folder_path, submission_file)
                dst_path = os.path.join(folder_path, 'sample_submission.csv')
                shutil.move(src_path, dst_path)
                print(f"Renamed {submission_file} to sample_submission.csv in {folder}")

def copy_answer_files(answers_folders: List[str], data_resplit_dir: str, answers_dir: str) -> None:
    """Copy test_answer.csv files from answers to data_resplit folders"""
    for folder in answers_folders:
        answer_folder_path = os.path.join(answers_dir, folder)
        test_answer_path = os.path.join(answer_folder_path, 'test_answer.csv')
        
        assert os.path.exists(test_answer_path), f"test_answer.csv does not exist in {folder} folder in answers directory"
        
        data_resplit_folder_path = os.path.join(data_resplit_dir, folder)
        if os.path.exists(data_resplit_folder_path):
            shutil.copy2(test_answer_path, data_resplit_folder_path)
            print(f"Copied test_answer.csv from answers/{folder} to data_resplit/{folder}")

def copy_task_files(task_files: List[str], data_resplit_dir: str, task_dir: str) -> None:
    """Copy task description files to data_resplit folders"""
    for task_file in task_files:
        task_file_path = os.path.join(task_dir, f"{task_file}.txt")
        data_resplit_folder_path = os.path.join(data_resplit_dir, task_file)
        
        if os.path.exists(data_resplit_folder_path):
            shutil.copy2(task_file_path, data_resplit_folder_path)
            print(f"Copied {task_file}.txt from task directory to data_resplit/{task_file}")

def verify_required_files(folders: List[str], data_resplit_dir: str) -> None:
    """Verify that all required files exist and rename description files"""
    for folder in folders:
        folder_path = os.path.join(data_resplit_dir, folder)
        files = os.listdir(folder_path)
        
        # Check if required files exist
        required_files = ['test_answer.csv', 'test.csv', 'train.csv', 'sample_submission.csv', f"{folder}.txt"]
        for required_file in required_files:
            if required_file not in files:
                print(f"Error: Required file {required_file} does not exist in {folder}")
                sys.exit(1)
        
        # Check if there are any extra files
        allowed_files = set(required_files)
        for file in files:
            if file not in allowed_files:
                print(f"Error: Extra file {file} found in {folder}")
                sys.exit(1)
        
        # Rename the txt file to description.txt
        txt_file_path = os.path.join(folder_path, f"{folder}.txt")
        description_file_path = os.path.join(folder_path, "description.txt")
        os.rename(txt_file_path, description_file_path)
        print(f"Renamed {folder}.txt to description.txt in {folder}")

def check_column_counts(folders: List[str], data_resplit_dir: str) -> dict:
    """Check if sample_submission.csv and test_answer.csv have the same number of columns"""
    print("\nChecking if sample_submission.csv and test_answer.csv have the same number of columns in each folder...")
    
    column_count = {}
    
    for folder in folders:
        folder_path = os.path.join(data_resplit_dir, folder)
        
        sample_submission_path = os.path.join(folder_path, 'sample_submission.csv')
        test_answer_path = os.path.join(folder_path, 'test_answer.csv')
        
        # Read the first line of each file to get the headers
        with open(sample_submission_path, 'r') as f:
            sample_submission_header = f.readline().strip().split(',')
        
        with open(test_answer_path, 'r') as f:
            test_answer_header = f.readline().strip().split(',')
        
        # Compare the number of columns
        sample_submission_cols = len(sample_submission_header)
        test_answer_cols = len(test_answer_header)
        
        if sample_submission_cols == test_answer_cols:
            if sample_submission_cols not in column_count:
                column_count[sample_submission_cols] = []
            column_count[sample_submission_cols].append(folder)
        else:
            print(f"Mismatch in {folder}: sample_submission.csv has {sample_submission_cols} columns, test_answer.csv has {test_answer_cols} columns")
    
    # Print the statistics
    print("\nStatistics of folders by number of columns:")
    for cols, folders in sorted(column_count.items()):
        print(f"{cols} column(s): {len(folders)} folder(s)")
    
    # Print folders with more than 2 columns
    for cols, folders in sorted(column_count.items()):
        if cols > 2:
            print(f"Columns: {cols}  Folders: {', '.join(folders)}")
    
    return column_count

def modify_submission_files(folders: List[str], data_resplit_dir: str) -> None:
    """Modify sample_submission.csv files to match test_answer.csv format"""
    print("\nModifying sample_submission.csv files to match test_answer.csv format...")
    
    for folder in folders:
        folder_path = os.path.join(data_resplit_dir, folder)
        
        sample_submission_path = os.path.join(folder_path, 'sample_submission.csv')
        test_answer_path = os.path.join(folder_path, 'test_answer.csv')
        
        test_answer_df = pd.read_csv(test_answer_path)
        sample_submission_df = pd.read_csv(sample_submission_path)
        
        # Check if the files have the same structure
        if list(test_answer_df.columns) == list(sample_submission_df.columns):
            print(f"Modifying {folder}/sample_submission.csv to match test_answer.csv format")
            
            # Create a new sample_submission based on test_answer structure
            sample_submission_df = test_answer_df.copy()
            
            # Randomly select a row and apply its values to all rows (excluding first column)
            random_row_idx = np.random.randint(0, len(sample_submission_df))
            random_row = sample_submission_df.iloc[random_row_idx, 1:].values
            
            for i in range(len(sample_submission_df)):
                sample_submission_df.iloc[i, 1:] = random_row
            
            # Save the modified sample_submission.csv
            sample_submission_df.to_csv(sample_submission_path, index=False)

def verify_file_structure(folders: List[str], data_resplit_dir: str) -> None:
    """Verify that sample_submission.csv and test_answer.csv have the same structure"""
    print("\nVerifying sample_submission.csv and test_answer.csv structure...")
    
    for folder in folders:
        folder_path = os.path.join(data_resplit_dir, folder)
        
        sample_submission_path = os.path.join(folder_path, 'sample_submission.csv')
        test_answer_path = os.path.join(folder_path, 'test_answer.csv')
        
        sample_submission_df = pd.read_csv(sample_submission_path)
        test_answer_df = pd.read_csv(test_answer_path)
        
        # Check structural similarities
        same_columns = len(sample_submission_df.columns) == len(test_answer_df.columns)
        same_rows = len(sample_submission_df) == len(test_answer_df)
        first_col_identical = sample_submission_df.iloc[:, 0].equals(test_answer_df.iloc[:, 0])
        
        # Check data types
        same_dtypes = True
        if len(sample_submission_df.columns) > 1:
            for i in range(1, len(sample_submission_df.columns)):
                if sample_submission_df.iloc[:, i].dtype != test_answer_df.iloc[:, i].dtype:
                    same_dtypes = False
                    break
        
        if not (same_columns and same_rows and first_col_identical and same_dtypes):
            print(f"  WARNING: Files in {folder} have structural differences!")
    
    print("\nVerification complete.")

def copy_to_competition_structure(folders: List[str], data_resplit_dir: str, competition_dir: str) -> None:
    """Copy files to competition directory structure"""
    print("\nCopying files to competition directory structure...")
    
    for folder in folders:
        # Source folder
        source_folder = os.path.join(data_resplit_dir, folder)
        
        # Destination folders
        public_dest_folder = os.path.join(competition_dir, folder, "data", "public")
        private_dest_folder = os.path.join(competition_dir, folder, "data", "private")
        
        # Create destination directories
        os.makedirs(public_dest_folder, exist_ok=True)
        os.makedirs(private_dest_folder, exist_ok=True)
        
        # Copy public files
        public_files = ["description.txt", "train.csv", "test.csv", "sample_submission.csv"]
        for file in public_files:
            source_file = os.path.join(source_folder, file)
            if os.path.exists(source_file):
                shutil.copy2(source_file, os.path.join(public_dest_folder, file))
                print(f"  Copied {file} to {public_dest_folder}")
            else:
                print(f"  Warning: {file} not found in {source_folder}")
        
        # Copy private file
        private_file = "test_answer.csv"
        source_file = os.path.join(source_folder, private_file)
        if os.path.exists(source_file):
            shutil.copy2(source_file, os.path.join(private_dest_folder, private_file))
            print(f"  Copied {private_file} to {private_dest_folder}")
        else:
            print(f"  Warning: {private_file} not found in {source_folder}")
    
    print("File copying complete.")

def organize_competition_folders(competition_base_dir: str) -> None:
    """Organize competition folders"""
    print("\nOrganizing competition folders...")
    
    # Create competition directory if it doesn't exist
    os.makedirs(competition_base_dir, exist_ok=True)
    
    # Get list of competition folders
    competition_folders = [f for f in os.listdir(competition_base_dir) 
                          if os.path.isdir(os.path.join(competition_base_dir, f))]
    
    for folder in competition_folders:
        # Define paths
        comp_folder_path = os.path.join(competition_base_dir, folder)
        info_dir = os.path.join(comp_folder_path, "info")
        private_data_dir = os.path.join(comp_folder_path, "data", "private")
        
        # Check if info directory exists
        if os.path.exists(info_dir) and os.path.exists(private_data_dir):
            # Check for leaderboard files
            public_leaderboard = os.path.join(info_dir, "public_leaderboard.csv")
            private_leaderboard = os.path.join(info_dir, "private_leaderboard.csv")
            
            # Move public_leaderboard.csv if it exists
            if os.path.exists(public_leaderboard):
                shutil.move(public_leaderboard, os.path.join(private_data_dir, "public_leaderboard.csv"))
                print(f"  Moved public_leaderboard.csv from {info_dir} to {private_data_dir}")
            
            # Move private_leaderboard.csv if it exists
            if os.path.exists(private_leaderboard):
                shutil.move(private_leaderboard, os.path.join(private_data_dir, "private_leaderboard.csv"))
                print(f"  Moved private_leaderboard.csv from {info_dir} to {private_data_dir}")
    
    print("Competition folders organization complete.")
    
def copy_leaderboard_files(source_base_dir: str, target_base_dir: str) -> None:
    """
    Copy leaderboard files from source directory to target directory.
    
    For each folder in target_base_dir, find the same folder name in source_base_dir,
    and copy info/private_leaderboard.csv and info/public_leaderboard.csv to 
    target_folder/data/private/.
    
    Args:
        source_base_dir: Base directory containing source folders with leaderboard files
        target_base_dir: Base directory containing target folders to copy leaderboard files to
    """
    print("\nCopying leaderboard files...")
    
    # Check if source directory exists
    if not os.path.exists(source_base_dir):
        print(f"Source directory {source_base_dir} does not exist.")
        return # Return early if source doesn't exist
    
    # Check if target directory exists
    if not os.path.exists(target_base_dir):
        print(f"Target directory {target_base_dir} does not exist.")
        return # Return early if target doesn't exist
    
    # Get list of folders in target directory
    target_folders = [f for f in os.listdir(target_base_dir) 
                     if os.path.isdir(os.path.join(target_base_dir, f))]
    
    # Get list of folders in source directory
    source_folders = [f for f in os.listdir(source_base_dir) 
                     if os.path.isdir(os.path.join(source_base_dir, f))]
    
    # Count successful copies
    copied_count = 0
    
    for folder in target_folders:
        # Check if folder exists in source directory
        if folder in source_folders:
            # Define paths
            source_info_dir = os.path.join(source_base_dir, folder, "info")
            target_private_dir = os.path.join(target_base_dir, folder, "data", "private")
            
            # Create target directory if it doesn't exist
            os.makedirs(target_private_dir, exist_ok=True)
            
            # Check if source info directory exists
            if os.path.exists(source_info_dir):
                # Define leaderboard file paths
                public_leaderboard = os.path.join(source_info_dir, "public_leaderboard.csv")
                private_leaderboard = os.path.join(source_info_dir, "private_leaderboard.csv")
                
                # Copy public_leaderboard.csv if it exists
                if os.path.exists(public_leaderboard):
                    shutil.copy2(public_leaderboard, os.path.join(target_private_dir, "public_leaderboard.csv"))
                    print(f"  Copied public_leaderboard.csv from {source_info_dir} to {target_private_dir}")
                    copied_count += 1
                else:
                    print(f"  Warning: public_leaderboard.csv not found in {source_info_dir}")
                
                # Copy private_leaderboard.csv if it exists
                if os.path.exists(private_leaderboard):
                    shutil.copy2(private_leaderboard, os.path.join(target_private_dir, "private_leaderboard.csv"))
                    # print(f"  Copied private_leaderboard.csv from {source_info_dir} to {target_private_dir}") # Keep output concise
                    copied_count += 1
                else:
                    print(f"  Warning: private_leaderboard.csv not found in {source_info_dir}")
            else:
                print(f"  Warning: Info directory not found for {folder} in source directory: {source_info_dir}") # Improved warning
        else:
            print(f"  Warning: Folder {folder} not found in source directory {source_base_dir}") # Improved warning
    
    print(f"Leaderboard file copying complete. Copied {copied_count} files.")


def main():
    args = parse_args()
    # Configurable paths - convert to absolute paths immediately
    raw_dir = os.path.abspath(args.raw_dir)
    prepared_dir = os.path.abspath(args.prepared_dir)

    # Define data directory paths
    data_dir = os.path.join(raw_dir, "data")

    # Determine the script's directory and the leaderboard source directory relative to it
    script_path = Path(__file__).resolve() # Get absolute path of the script
    script_dir = script_path.parent # Get directory containing the script
    project_root = script_dir.parent # Assume script is in 'prepare', project root is one level up
    leaderboard_source_dir = project_root / "mledojo" / "competitions"

    
    # Step 1: Download and extract data
    # Ensure raw_dir exists before downloading
    os.makedirs(raw_dir, exist_ok=True)
    download_and_extract_data(raw_dir)
    
    # Step 2: Get folder lists
    data_resplit_folders, answers_folders, task_files, data_resplit_dir, answers_dir, task_dir = get_folder_lists(data_dir)
    
    # Step 3: Fix submission files
    fix_submission_files(data_resplit_folders, data_resplit_dir)
    
    # Step 4: Copy answer files
    copy_answer_files(answers_folders, data_resplit_dir, answers_dir)
    
    # Step 5: Copy task files
    copy_task_files(task_files, data_resplit_dir, task_dir)
    
    # Step 6: Verify required files
    verify_required_files(data_resplit_folders, data_resplit_dir)
    print("All data_resplit folders have the required files and only those files.")
    
    # Step 7: Check column counts
    column_count = check_column_counts(data_resplit_folders, data_resplit_dir)
    
    # Step 8: Modify submission files
    modify_submission_files(data_resplit_folders, data_resplit_dir)
    print("Sample submission files have been modified to match test_answer.csv format.")
    
    # Step 9: Verify file structure
    verify_file_structure(data_resplit_folders, data_resplit_dir)

    # Step 10: Copy to competition structure
    # Ensure prepared_dir exists before copying
    os.makedirs(prepared_dir, exist_ok=True)
    copy_to_competition_structure(data_resplit_folders, data_resplit_dir, prepared_dir)

    # Step 11: Copy leaderboard files
    copy_leaderboard_files(str(leaderboard_source_dir), prepared_dir) # Pass absolute path
    
    # Step 12: Remove data folder
    # Check if data_dir exists before removing
    if os.path.exists(data_dir) and os.path.isdir(data_dir):
         print(f"\nRemoving temporary data directory: {data_dir}")
         shutil.rmtree(data_dir) # Use shutil.rmtree for better cross-platform compatibility
    else:
         print(f"\nTemporary data directory not found or already removed: {data_dir}")

    # Step 13: Remove raw_dir if it's empty and different from prepared_dir
    if raw_dir != prepared_dir and os.path.exists(raw_dir) and not os.listdir(raw_dir):
        print(f"Removing empty raw directory: {raw_dir}")
        os.rmdir(raw_dir) # Remove empty directory

if __name__ == "__main__":
    main()