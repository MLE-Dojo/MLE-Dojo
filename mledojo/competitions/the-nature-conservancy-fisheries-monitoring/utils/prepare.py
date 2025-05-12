#!/usr/bin/env python3
import os
import shutil
import random
import csv
from pathlib import Path
from tqdm import tqdm

# Full classes in the competition submission (order matters)
FULL_CLASSES = ["ALB", "BET", "DOL", "LAG", "NoF", "OTHER", "SHARK", "YFT"]

def prepare(raw: Path, public: Path, private: Path):
    """
    Prepare data for the-nature-conservancy-fisheries-monitoring competition.
    
    Args:
        raw: Path to raw data directory
        public: Path to public data directory
        private: Path to private data directory
    """
    os.system("unzip -o " + str(raw / "train.zip") + " -d " + str(raw))
    # Create destination directories
    train_dir = public / "train"
    test_dir = public / "test"
    for d in [private, public, train_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Output file paths
    test_answer_csv = private / "test_answer.csv"
    sample_submission_csv = public / "sample_submission.csv"
    
    # Split ratio (80% train, 20% test)
    train_ratio = 0.8
    
    # Process training set (splitting into local train and test)
    src_train_dir = raw / "train"
    test_gt = {}  # Mapping: new_test_filename -> class label
    
    # For assertion: store counts per class
    orig_counts = {}
    copied_train_counts = {}
    copied_test_counts = {}
    
    # Iterate through each class folder in the original train directory
    for class_folder in tqdm(os.listdir(src_train_dir), desc="Processing Classes"):
        class_src_path = src_train_dir / class_folder
        # Skip if not a directory or if it's an OS-specific folder/file
        if not class_src_path.is_dir() or class_folder.startswith('.') or class_folder.startswith('__MACOSX'):
            continue
            
        # List image files in the class folder; filter out OS metadata
        all_files = [f for f in os.listdir(class_src_path) 
                    if not (f.startswith('.') or f.startswith('__MACOSX'))]
        orig_counts[class_folder] = len(all_files)
        
        # Shuffle files to randomize split
        random.shuffle(all_files)
        
        n_total = len(all_files)
        n_train = int(n_total * train_ratio)
        
        train_files = all_files[:n_train]
        test_files = all_files[n_train:]
        
        # Create destination subdirectory for training images for this class
        dest_train_class = train_dir / class_folder
        dest_train_class.mkdir(exist_ok=True)
        
        # Copy training images
        copied_train = 0
        for f in train_files:
            src_file = class_src_path / f
            dest_file = dest_train_class / f
            shutil.copy2(src_file, dest_file)
            copied_train += 1
        copied_train_counts[class_folder] = copied_train
        
        # For test images, we copy them into a flattened test folder without class name prefix
        copied_test = 0
        for f in test_files:
            src_file = class_src_path / f
            dest_file = test_dir / f
            shutil.copy2(src_file, dest_file)
            test_gt[f] = class_folder
            copied_test += 1
        copied_test_counts[class_folder] = copied_test
    
    # Assertion: Check that the sum of copied training and test images equals original count per class
    for cls in orig_counts:
        total_copied = copied_train_counts.get(cls, 0) + copied_test_counts.get(cls, 0)
        assert total_copied == orig_counts[cls], f"Mismatch count for class {cls}: original {orig_counts[cls]}, copied {total_copied}"
    
    # Create test_answer.csv in the private folder
    with open(test_answer_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["image"] + FULL_CLASSES
        writer.writerow(header)
        for img_filename, label in sorted(test_gt.items()):
            row = [img_filename] + one_hot_vector(label)
            writer.writerow(row)
    
    # Create sample_submission.csv in the public folder
    with open(sample_submission_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["image"] + FULL_CLASSES
        writer.writerow(header)
        dummy_row = dummy_prob_vector()
        # Read test_answer.csv to get the same order of images for submission
        test_images = []
        with open(test_answer_csv, "r") as ansfile:
            reader = csv.reader(ansfile)
            next(reader)  # Skip header
            for row in reader:
                test_images.append(row[0])
        
        for img in test_images:
            writer.writerow([img] + dummy_row)
    
    
    # Final Assertions for CSV files
    verify_csv_files(test_answer_csv, sample_submission_csv, test_dir, test_gt)
    
    print("Data reorganization complete. All assertions passed.")

    # Clean up
    shutil.rmtree(raw)

def one_hot_vector(label):
    """Create one-hot encoding for FULL_CLASSES based on the label."""
    return ["1" if cls == label else "0" for cls in FULL_CLASSES]

def dummy_prob_vector():
    """Create dummy probabilities: uniform probability 1/8 for each class."""
    prob = 1.0 / len(FULL_CLASSES)
    return [f"{prob:.3f}" for _ in FULL_CLASSES]

def verify_csv_files(test_answer_csv, sample_submission_csv, test_dir, test_gt):
    """Verify the correctness of generated CSV files."""
    # 1. Verify header of test_answer.csv and sample_submission.csv matches exactly
    expected_header = ["image"] + FULL_CLASSES
    
    def verify_csv_header(csv_path, expected_header):
        with open(csv_path, "r") as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            assert header == expected_header, f"Header mismatch in {csv_path}"
    
    verify_csv_header(test_answer_csv, expected_header)
    verify_csv_header(sample_submission_csv, expected_header)
    
    # 2. Verify that the number of rows in test_answer.csv equals number of files in test_dir
    num_csv_rows = 0
    with open(test_answer_csv, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip header
        for row in reader:
            num_csv_rows += 1
    
    num_test_files = len([f for f in os.listdir(test_dir) 
                         if not (f.startswith('.') or f.startswith('__MACOSX'))])
    assert num_csv_rows == num_test_files, f"Row count in test_answer.csv ({num_csv_rows}) does not match test folder file count ({num_test_files})"
    
    # 3. Verify that each image in sample_submission.csv matches the test_answer.csv
    submission_images = []
    with open(sample_submission_csv, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            submission_images.append(row[0])
    
    assert sorted(submission_images) == sorted(list(test_gt.keys())), "Mismatch between sample_submission and test_answer image list"