#!/usr/bin/env python3
"""
This script reorganizes the Yelp dataset by:
  - Splitting the original training reviews into a local training set and test set
  - Removing the "stars" field from test reviews and saving it separately
  - Copying supporting files (business, user, checkin) unmodified
  - Generating a sample_submission.csv with dummy predictions
"""

import os
import json
import random
import csv
import shutil
from pathlib import Path
from tqdm import tqdm

def prepare(raw: Path, public: Path, private: Path):
    """
    Prepare data for yelp-recsys-2013 competition.
    
    Args:
        raw: Path to raw data directory
        public: Path to public data directory
        private: Path to private data directory
    """
    # Create destination directories
    os.system("unzip -o " + str(raw / "yelp_test_set.zip") + " -d " + str(raw))
    os.system("unzip -o " + str(raw / "final_test_set.zip") + " -d " + str(raw))
    os.system("unzip -o " + str(raw / "yelp_training_set.zip") + " -d " + str(raw))
    os.system("unzip -o " + str(raw / "yelp_training_set_mac.zip") + " -d " + str(raw))
    train_dir = public / "train"
    test_dir = public / "test"
    
    for d in [private, public, train_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Fixed random seed for reproducibility
    RANDOM_SEED = 42
    
    # Paths to original files
    orig_train_review = raw / "yelp_training_set" / "yelp_training_set_review.json"
    orig_test_review = raw / "yelp_test_set" / "yelp_test_set_review.json"
    orig_train_business = raw / "yelp_training_set" / "yelp_training_set_business.json"
    orig_train_user = raw / "yelp_training_set" / "yelp_training_set_user.json"
    orig_train_checkin = raw / "yelp_training_set" / "yelp_training_set_checkin.json"
    orig_test_business = raw / "yelp_test_set" / "yelp_test_set_business.json"
    orig_test_user = raw / "yelp_test_set" / "yelp_test_set_user.json"
    orig_test_checkin = raw / "yelp_test_set" / "yelp_test_set_checkin.json"
    
    # Output paths
    train_review_out = train_dir / "yelp_training_set_review.json"
    test_review_out = test_dir / "yelp_test_set_review.json"
    train_business_out = train_dir / "yelp_training_set_business.json"
    test_business_out = test_dir / "yelp_test_set_business.json"
    train_user_out = train_dir / "yelp_training_set_user.json"
    test_user_out = test_dir / "yelp_test_set_user.json"
    train_checkin_out = train_dir / "yelp_training_set_checkin.json"
    test_checkin_out = test_dir / "yelp_test_set_checkin.json"
    test_answer_path = private / "test_answer.csv"
    sample_submission_path = public / "sample_submission.csv"
    
    # Step 1: Count reviews in original files
    print("Counting reviews in original files...")
    total_reviews = count_lines(orig_train_review)
    num_test_orig = count_lines(orig_test_review)
    
    assert total_reviews >= num_test_orig, "The number of reviews in training must be >= number in test."
    
    # Step 2: Read all reviews from the original training review file
    reviews = []
    with open(orig_train_review, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_reviews, desc="Reading training reviews"):
            line = line.strip()
            if line:
                reviews.append(line)
    
    # Step 3: Split into training and test sets
    indices = list(range(total_reviews))
    random.seed(RANDOM_SEED)
    test_indices = sorted(random.sample(indices, num_test_orig))
    test_set = [reviews[i] for i in test_indices]
    training_set = [reviews[i] for i in range(total_reviews) if i not in set(test_indices)]
    
    # Verify counts
    assert len(test_set) == num_test_orig, "Test review count in split does not equal original test review count."
    assert (len(training_set) + len(test_set)) == total_reviews, "Total reviews after split mismatch."
    
    # Step 4: Write training reviews
    with open(train_review_out, "w", encoding="utf-8") as fout:
        for line in tqdm(training_set, desc="Writing training reviews"):
            fout.write(line.rstrip('\n') + "\n")
    
    # Step 5: Process test reviews - create test features and answers
    with open(test_review_out, "w", encoding="utf-8") as fout_test, \
         open(test_answer_path, "w", newline='', encoding="utf-8") as fout_ans:
        
        csv_writer = csv.writer(fout_ans)
        csv_writer.writerow(["RecommendationId", "Stars"])
        
        rec_id = 1
        for line in tqdm(test_set, desc="Processing test reviews"):
            review_obj = json.loads(line)
            stars = review_obj.pop("stars", None)
            assert stars is not None, f"Review missing 'stars' field: {review_obj}"
            fout_test.write(json.dumps(review_obj) + "\n")
            csv_writer.writerow([rec_id, stars])
            rec_id += 1
    
    # Step 6: Process business files
    print("Processing business files...")
    # Copy training business file as is
    shutil.copyfile(orig_train_business, train_business_out)
    
    # Process test business file - remove stars field
    with open(orig_test_business, "r", encoding="utf-8") as fin, \
         open(test_business_out, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="Processing test business data"):
            if line.strip():
                business_obj = json.loads(line)
                business_obj.pop("stars", None)  # Remove stars field from test businesses
                fout.write(json.dumps(business_obj) + "\n")
    
    # Step 7: Process user files
    print("Processing user files...")
    # Copy training user file as is
    shutil.copyfile(orig_train_user, train_user_out)
    
    # Process test user file - remove average_stars and votes fields
    with open(orig_test_user, "r", encoding="utf-8") as fin, \
         open(test_user_out, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="Processing test user data"):
            if line.strip():
                user_obj = json.loads(line)
                user_obj.pop("average_stars", None)  # Remove average_stars field
                user_obj.pop("votes", None)  # Remove votes field
                fout.write(json.dumps(user_obj) + "\n")
    
    # Step 8: Copy checkin files (format is the same for both training and test)
    print("Copying checkin files...")
    shutil.copyfile(orig_train_checkin, train_checkin_out)
    shutil.copyfile(orig_test_checkin, test_checkin_out)
    
    # Step 9: Create sample submission file
    recommendation_ids = []
    with open(test_answer_path, "r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            recommendation_ids.append(row["RecommendationId"])
    
    with open(sample_submission_path, "w", newline='', encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["RecommendationId", "Stars"])
        for rec_id in recommendation_ids:
            writer.writerow([rec_id, 3.5])  # Dummy prediction value
    
    print("Data preparation complete.")
    
    # Clean up
    shutil.rmtree(raw)

def count_lines(file_path):
    """Count the number of non-empty lines in a file."""
    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count