#!/usr/bin/env python3
import os
import shutil
import random
import math
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def prepare(raw: Path, public: Path, private: Path):
    
    # Set fixed random seed for reproducibility.
    random.seed(42)

    # Paths for original data
    # DATA_DIR = "data"
    DATA_DIR = raw
    TRAIN_DIR = os.path.join(DATA_DIR, "train_v2")
    TRAIN_SEGMENTATION_FILE = os.path.join(DATA_DIR, "train_ship_segmentations_v2.csv")
    ZIP_FILE = os.path.join(DATA_DIR, "airbus-ship-detection.zip")
    # (Note: test_v2 and sample_submission_v2.csv are not processed)

    # Paths for organized data
    # ORGANIZED_DIR = "organized_data"
    ORGANIZED_DIR = Path(str(raw).replace("raw", "data"))
    PUBLIC_DIR = os.path.join(ORGANIZED_DIR, "public")
    PRIVATE_DIR = os.path.join(ORGANIZED_DIR, "private")
    PUBLIC_TRAIN_DIR = os.path.join(PUBLIC_DIR, "train")
    PUBLIC_TEST_DIR = os.path.join(PUBLIC_DIR, "test")

    # Create directory structure
    for directory in [PUBLIC_TRAIN_DIR, PUBLIC_TEST_DIR, PRIVATE_DIR]:
        os.makedirs(directory, exist_ok=True)

    # 1. List image file names from data/train_v2
    image_files = [f for f in os.listdir(TRAIN_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    n_total = len(image_files)
    if n_total == 0:
        raise ValueError("No image files found in {}".format(TRAIN_DIR))

    # Calculate test_size using the given ratio: 15601 / (15601+192551)
    ratio = 15601 / (15601 + 192551)
    test_size = round(n_total * ratio)
    train_size = n_total - test_size

    # Randomly split the image file names with fixed seed
    shuffled_images = image_files.copy()
    random.shuffle(shuffled_images)
    test_images = shuffled_images[:test_size]
    train_images = shuffled_images[test_size:]

    # 2. Copy images to organized_data/public/train and organized_data/public/test
    print("Copying training images...")
    for img in tqdm(train_images, desc="Copying train images"):
        src = os.path.join(TRAIN_DIR, img)
        dst = os.path.join(PUBLIC_TRAIN_DIR, img)
        shutil.copy2(src, dst)

    print("Copying test images...")
    for img in tqdm(test_images, desc="Copying test images"):
        src = os.path.join(TRAIN_DIR, img)
        dst = os.path.join(PUBLIC_TEST_DIR, img)
        shutil.copy2(src, dst)

    # 3. Process ground truth CSV file to create test_answer.csv.
    # Read the original CSV.
    gt_df = pd.read_csv(TRAIN_SEGMENTATION_FILE)

    # Create a dictionary mapping image id to its EncodedPixels list from original.
    # Here, we group by image id and if there are multiple entries join them with a space.
    grouped = gt_df.groupby("ImageId")["EncodedPixels"].apply(list).to_dict()

    # Prepare test answer rows ensuring each test image has exactly one row.
    test_answer_rows = []
    for img in test_images:
        # Remove extension for matching if needed. But in our CSV, ImageId should be exactly the file name.
        # We assume the CSV "ImageId" column has the same file names as in train_v2.
        encodings = grouped.get(img, [])
        # Remove any null/NaN entries.
        encodings = [str(e) for e in encodings if pd.notna(e)]
        if encodings:
            # If multiple encodings exist, join them with a space.
            encoded_str = " ".join(encodings)
        else:
            encoded_str = ""
        test_answer_rows.append({"ImageId": img, "EncodedPixels": encoded_str})

    # Create DataFrame for test_answer.csv with header "ImageId,EncodedPixels"
    test_answer_df = pd.DataFrame(test_answer_rows, columns=["ImageId", "EncodedPixels"])
    TEST_ANSWER_FILE = os.path.join(PRIVATE_DIR, "test_answer.csv")
    test_answer_df.to_csv(TEST_ANSWER_FILE, index=False)

    # 4. Create sample_submission.csv in organized_data/public/
    # The sample submission should have the same header and same ordering of image IDs as in test_answer.csv.
    sample_submission_df = test_answer_df.copy()
    # For dummy answers, use right typed dummy value, here we use an empty string.
    sample_submission_df["EncodedPixels"] = ""  
    SAMPLE_SUBMISSION_FILE = os.path.join(PUBLIC_DIR, "sample_submission.csv")
    sample_submission_df.to_csv(SAMPLE_SUBMISSION_FILE, index=False)

    # 5. Copy airbus-ship-detection.zip into organized_data/public/
    destination_zip = os.path.join(PUBLIC_DIR, os.path.basename(ZIP_FILE))
    shutil.copy2(ZIP_FILE, destination_zip)

    # 6. Copy original train_ship_segmentations_v2.csv into organized_data/public/
    destination_train_seg = os.path.join(PUBLIC_DIR, os.path.basename(TRAIN_SEGMENTATION_FILE))
    shutil.copy2(TRAIN_SEGMENTATION_FILE, destination_train_seg)

    # 7. Run assertions to confirm correctness.

    # Count assertions: Verify total number of images copied equals original count.
    num_train_copied = len(os.listdir(PUBLIC_TRAIN_DIR))
    num_test_copied = len(os.listdir(PUBLIC_TEST_DIR))
    assert num_train_copied + num_test_copied == n_total, \
        "Total images in public/train and public/test do not equal the original count."

    # Assertion for test size ratio (approximately 7.5%)
    assert num_test_copied == test_size, \
        "Number of test images ({}) does not equal expected test size ({}).".format(num_test_copied, test_size)

    # Check that each test image from test set appears exactly once in test folder.
    copied_test_images = sorted(os.listdir(PUBLIC_TEST_DIR))
    assert sorted(test_images) == copied_test_images, \
        "Mismatch between test images list and images in organized_data/public/test folder."

    # Check that test_answer.csv has the same number of rows as the number of test images.
    test_answer_df_loaded = pd.read_csv(TEST_ANSWER_FILE)
    assert len(test_answer_df_loaded) == num_test_copied, \
        "Number of rows in test_answer.csv does not equal number of test images."

    # Check that the header of test_answer.csv is exactly "ImageId,EncodedPixels"
    assert list(test_answer_df_loaded.columns) == ["ImageId", "EncodedPixels"], \
        "Header of test_answer.csv is incorrect."

    # Check that sample_submission.csv has the same number of rows and identical ImageId order.
    submission_df_loaded = pd.read_csv(SAMPLE_SUBMISSION_FILE)
    assert len(submission_df_loaded) == num_test_copied, \
        "Number of rows in sample_submission.csv does not equal number of test images."
    assert list(submission_df_loaded.columns) == ["ImageId", "EncodedPixels"], \
        "Header of sample_submission.csv is incorrect."
    assert list(submission_df_loaded["ImageId"]) == list(test_answer_df_loaded["ImageId"]), \
        "ImageId order in sample_submission.csv does not match test_answer.csv."

    print("All assertions passed. Data has been organized successfully.")

    # Clean up
    shutil.rmtree(DATA_DIR)
    # Delete zip file of public/
    for file in os.listdir(PUBLIC_DIR):
        if file.endswith('.zip'):
            os.remove(os.path.join(PUBLIC_DIR, file))
