import os
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
import ast
from pathlib import Path

def prepare(raw: Path, public: Path, private: Path):
    """
    Prepares data for the Global Wheat Detection competition by:
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
    (public / "train").mkdir(exist_ok=True, parents=True)
    (public / "test").mkdir(exist_ok=True, parents=True)
    print("Created output directories")
    
    # Read the original train and test files
    train_df = pd.read_csv(raw / "train.csv")
    
    # Count the number of unique images in training data
    unique_train_images = train_df['image_id'].unique()
    num_train_images = len(unique_train_images)
    
    # Count the number of test images
    test_images = os.listdir(raw / "test")
    num_test_images = len(test_images)
    
    print(f"Original dataset: {num_train_images} unique training images, {num_test_images} test images")
    
    # Calculate the split ratio based on original data
    try:
        test_ratio = num_test_images / (num_train_images + num_test_images)
        print(f"Calculated test ratio: {test_ratio}")
        if test_ratio > 1 or test_ratio < 0.05:
            print(f"Calculated test ratio {test_ratio} is improper, using default value of 0.2")
            test_ratio = 0.2
    except Exception as e:
        print(f"Error calculating test ratio: {e}, using default value of 0.2")
        test_ratio = 0.2
    
    print(f"Using test ratio: {test_ratio}")
    
    # Split the unique image IDs into train and test sets
    train_image_ids, test_image_ids = train_test_split(
        unique_train_images, 
        test_size=test_ratio,
        random_state=42
    )
    
    # Sort test_image_ids to ensure consistent ordering across all files
    test_image_ids = sorted(test_image_ids)
    
    print(f"Split data: {len(train_image_ids)} new training images, {len(test_image_ids)} new test images")
    
    # Verify the split is correct
    assert len(train_image_ids) + len(test_image_ids) == num_train_images, "Split sizes don't add up to original size"
    assert abs(len(test_image_ids) / num_train_images - test_ratio) < 0.01, "Test ratio is not as expected"
    
    # Ensure there's no overlap between train and test sets
    assert len(set(train_image_ids).intersection(set(test_image_ids))) == 0, "Train and test sets have overlapping image IDs"
    
    # Create new training and test dataframes
    new_train_df = train_df[train_df['image_id'].isin(train_image_ids)]
    new_test_df = train_df[train_df['image_id'].isin(test_image_ids)]
    
    # Group the test dataframe by image_id for creating test_answer.csv
    test_answers_grouped = new_test_df.groupby('image_id')
    
    # Create a test_images_df for easier handling
    test_images_info = []
    for image_id in test_image_ids:  # Use sorted test_image_ids to maintain order
        if image_id in test_answers_grouped.groups:
            group = test_answers_grouped.get_group(image_id)
            # Take the first row for width and height (should be the same for all rows with same image_id)
            width = group['width'].iloc[0]
            height = group['height'].iloc[0]
            # Get all bounding boxes for this image
            bboxes = group['bbox'].tolist()
            test_images_info.append({
                'image_id': image_id,
                'width': width,
                'height': height,
                'bboxes': bboxes
            })
    
    test_images_df = pd.DataFrame(test_images_info)
    
    # Save the new training data
    new_train_df.to_csv(public / "train.csv", index=False)
    print("Created new training file")
    
    # Create a test.csv with only the image_id, width, and height (no bboxes)
    # Use test_images_df to maintain the same order as test_image_ids
    test_df_columns = test_images_df[['image_id', 'width', 'height']]
    test_df_columns.to_csv(public / "test.csv", index=False)
    print("Created new test file")
    
    # Create sample_submission.csv with the same order as test_image_ids
    sample_submission = pd.DataFrame({
        'image_id': test_images_df['image_id'],
        'PredictionString': ['' for _ in range(len(test_images_df))]
    })
    sample_submission.to_csv(public / "sample_submission.csv", index=False)
    print("Created sample_submission.csv")
    
    # Create test_answer.csv 
    # Format bounding boxes as required for submission (confidence x y width height)
    def format_predictions(row):
        if isinstance(row['bboxes'], list) and len(row['bboxes']) > 0:
            predictions = []
            for bbox_str in row['bboxes']:
                # Parse the bbox string into a list
                if isinstance(bbox_str, str):
                    bbox = ast.literal_eval(bbox_str)
                else:
                    bbox = bbox_str
                # Format as "confidence x y width height"
                predictions.append(f"1.0 {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}")
            return ' '.join(predictions)
        return ''
    
    test_images_df['PredictionString'] = test_images_df.apply(format_predictions, axis=1)
    test_answer_df = test_images_df[['image_id', 'PredictionString']]
    test_answer_df.to_csv(private / "test_answer.csv", index=False)
    print("Created test_answer.csv")
    
    # Verify test_answer.csv and sample_submission.csv have the same columns
    sample_submission_cols = list(pd.read_csv(public / "sample_submission.csv").columns)
    test_answer_cols = list(pd.read_csv(private / "test_answer.csv").columns)
    assert sample_submission_cols == test_answer_cols, "Column mismatch between test_answer.csv and sample_submission.csv"
    print("Verified test_answer.csv and sample_submission.csv have the same columns")
    
    # Verify that test_answer.csv, sample_submission.csv, and test.csv have the same image_ids in the same order
    test_csv_ids = list(pd.read_csv(public / "test.csv")['image_id'])
    sample_submission_ids = list(pd.read_csv(public / "sample_submission.csv")['image_id'])
    test_answer_ids = list(pd.read_csv(private / "test_answer.csv")['image_id'])
    
    assert test_csv_ids == sample_submission_ids, "Image IDs in test.csv and sample_submission.csv don't match or have different order"
    assert test_csv_ids == test_answer_ids, "Image IDs in test.csv and test_answer.csv don't match or have different order"
    print("Verified all test image IDs match and have the same order across files")
    
    # Copy the image files
    print("Copying training images...")
    for image_id in train_image_ids:
        source_path = raw / "train" / f"{image_id}.jpg"
        if source_path.exists():
            shutil.copy(source_path, public / "train" / f"{image_id}.jpg")
    
    print("Copying test images...")
    for image_id in test_image_ids:
        source_path = raw / "train" / f"{image_id}.jpg"
        if source_path.exists():
            shutil.copy(source_path, public / "test" / f"{image_id}.jpg")
    
    # Verify that image files in test directory match the IDs in test.csv
    test_dir_images = sorted([f.stem for f in (public / "test").glob("*.jpg")])
    assert test_dir_images == test_csv_ids, "Image files in test directory don't match IDs in test.csv or have different order"
    print("Verified test image files match test.csv IDs and have the same order")
    
    
    print("Data preparation complete!")

    # clean up
    shutil.rmtree(raw)