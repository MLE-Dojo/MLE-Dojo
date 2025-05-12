import os
import shutil
import glob
import random
import csv
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
    # Fixed random seed for reproducibility
    random.seed(42)
    
    # Define directories
    train_tiff_dir = raw / "train" / "train"
    train_jpeg_dir = raw / "train_sm" / "train_sm"
    
    public_train_tiff_dir = public / "train"
    public_train_jpeg_dir = public / "train_sm"
    public_test_tiff_dir = public / "test"
    public_test_jpeg_dir = public / "test_sm"
    
    test_answer_csv = private / "test_answer.csv"
    sample_submission_csv = public / "sample_submission.csv"
    
    # Create directory structure
    for d in [public_train_tiff_dir, public_train_jpeg_dir, public_test_tiff_dir, public_test_jpeg_dir, private]:
        os.makedirs(d, exist_ok=True)
    
    # Helper function to get setId from filename
    def get_set_id(filename):
        base = os.path.basename(filename)
        return base.split("_")[0]
    
    # Group files by set
    def group_files_by_set(source_dir, ext):
        pattern = os.path.join(source_dir, f"*.{ext}")
        files = glob.glob(pattern)
        groups = {}
        for f in files:
            set_id = get_set_id(f)
            groups.setdefault(set_id, []).append(f)
        return groups
    
    # Group TIFF and JPEG files
    tiff_groups = group_files_by_set(train_tiff_dir, "tif")
    jpeg_groups = group_files_by_set(train_jpeg_dir, "jpeg")
    
    # Validate data integrity
    all_set_ids = list(tiff_groups.keys())
    for set_id in all_set_ids:
        assert len(tiff_groups[set_id]) == 5, f"Set {set_id} in TIFF folder does not have 5 files."
        assert set_id in jpeg_groups, f"Set {set_id} not found in JPEG folder."
        assert len(jpeg_groups[set_id]) == 5, f"Set {set_id} in JPEG folder does not have 5 files."
    
    # Split into train and test sets (56% train, 44% test)
    n_total = len(all_set_ids)
    train_count = round(n_total * 0.56)
    test_count = n_total - train_count
    
    random.shuffle(all_set_ids)
    train_set_ids = all_set_ids[:train_count]
    test_set_ids = all_set_ids[train_count:]
    
    # Copy files function
    def copy_files_for_sets(set_ids, source_groups_tiff, source_groups_jpeg,
                           target_tiff_dir, target_jpeg_dir):
        for set_id in tqdm(set_ids, desc="Copying sets"):
            # Copy TIFF files
            for file_path in source_groups_tiff[set_id]:
                target_path = os.path.join(target_tiff_dir, os.path.basename(file_path))
                shutil.copy2(file_path, target_path)
            # Copy JPEG files
            for file_path in source_groups_jpeg[set_id]:
                target_path = os.path.join(target_jpeg_dir, os.path.basename(file_path))
                shutil.copy2(file_path, target_path)
    
    # Copy train and test files
    copy_files_for_sets(train_set_ids, tiff_groups, jpeg_groups, public_train_tiff_dir, public_train_jpeg_dir)
    copy_files_for_sets(test_set_ids, tiff_groups, jpeg_groups, public_test_tiff_dir, public_test_jpeg_dir)
    
    # Verify file counts
    def assert_file_count(target_dir, expected_count, ext):
        files = glob.glob(os.path.join(target_dir, f"*.{ext}"))
        assert len(files) == expected_count * 5, f"Expected {expected_count*5} *.{ext} files in {target_dir}, found {len(files)}."
    
    assert_file_count(public_train_tiff_dir, len(train_set_ids), "tif")
    assert_file_count(public_train_jpeg_dir, len(train_set_ids), "jpeg")
    assert_file_count(public_test_tiff_dir, len(test_set_ids), "tif")
    assert_file_count(public_test_jpeg_dir, len(test_set_ids), "jpeg")
    
    # Create test_answer.csv with randomized orders
    with open(test_answer_csv, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["setId", "day"])
        for set_id in sorted(test_set_ids):
            # Generate random ordering for each set
            new_order = list(range(1, 6))
            random.shuffle(new_order)
            day_str = " ".join(map(str, new_order))
            writer.writerow([set_id, day_str])
            
            # Rename files according to new order
            temp_test_dir = str(public_test_tiff_dir) + "_temp"
            temp_test_sm_dir = str(public_test_jpeg_dir) + "_temp"
            os.makedirs(temp_test_dir, exist_ok=True)
            os.makedirs(temp_test_sm_dir, exist_ok=True)
            
            for old_num, new_num in enumerate(new_order, 1):
                # Handle .tif files
                old_tif = os.path.join(public_test_tiff_dir, f"{set_id}_{old_num}.tif")
                new_tif = os.path.join(temp_test_dir, f"{set_id}_{new_num}.tif")
                if os.path.exists(old_tif):
                    shutil.move(old_tif, new_tif)
                    
                # Handle .jpeg files
                old_jpeg = os.path.join(public_test_jpeg_dir, f"{set_id}_{old_num}.jpeg")
                new_jpeg = os.path.join(temp_test_sm_dir, f"{set_id}_{new_num}.jpeg")
                if os.path.exists(old_jpeg):
                    shutil.move(old_jpeg, new_jpeg)
            
            # Move files back from temp directories
            for filename in os.listdir(temp_test_dir):
                shutil.move(os.path.join(temp_test_dir, filename), os.path.join(public_test_tiff_dir, filename))
            for filename in os.listdir(temp_test_sm_dir):
                shutil.move(os.path.join(temp_test_sm_dir, filename), os.path.join(public_test_jpeg_dir, filename))
            
            # Clean up temp directories
            os.rmdir(temp_test_dir)
            os.rmdir(temp_test_sm_dir)
    
    # Verify test_answer.csv
    with open(test_answer_csv, mode="r") as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
        assert len(rows) == len(test_set_ids), "test_answer.csv row count mismatch."
        for row in rows:
            day_list = row["day"].split()
            assert len(day_list) == 5, f"Row for setId {row['setId']} does not contain 5 day values."
    
    # Create sample_submission.csv
    with open(sample_submission_csv, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["setId", "day"])
        for set_id in sorted(test_set_ids):
            # Dummy answer with random ordering
            numbers = list(range(1, 6))
            random.shuffle(numbers)
            day_str = " ".join(map(str, numbers))
            writer.writerow([set_id, day_str])
    
    # Verify sample_submission.csv
    with open(sample_submission_csv, mode="r") as csvfile:
        reader = csv.DictReader(csvfile)
        sample_rows = list(reader)
        assert reader.fieldnames == ["setId", "day"], "sample_submission.csv header mismatch."
        assert len(sample_rows) == len(test_set_ids), "sample_submission.csv row count mismatch."
        for row in sample_rows:
            day_list = row["day"].split()
            assert len(day_list) == 5, f"Row for setId {row['setId']} in sample_submission.csv does not contain 5 day values."
    
    print("Data preparation completed successfully.")

    # Clean up
    shutil.rmtree(raw)