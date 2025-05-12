import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from mledojo.competitions.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):
    # Read the train.csv file which contains image IDs and labels
    train_df = read_csv(raw / "train.csv")
    
    # Split the data into train and test sets
    # We use 0.1 ratio to avoid removing too many samples from train
    new_train, new_test = train_test_split(
        train_df, test_size=0.1, random_state=0
    )
    
    # Create sample submission file based on test set
    sample_submission = new_test.copy()
    sample_submission["label"] = 4  # Default prediction
    
    # Save CSV files
    new_train.to_csv(public / "train.csv", index=False)
    new_test.to_csv(private / "test_answer.csv", index=False)
    sample_submission.to_csv(public / "sample_submission.csv", index=False)
    
    # Copy train images
    (public / "train_images").mkdir(parents=True, exist_ok=True)
    for image_id in tqdm(new_train["image_id"], desc="Copying Train Images", total=len(new_train)):
        shutil.copy(raw / "train_images" / image_id, public / "train_images")
    
    # Copy test images
    (public / "test_images").mkdir(parents=True, exist_ok=True)
    for image_id in tqdm(new_test["image_id"], desc="Copying Test Images", total=len(new_test)):
        shutil.copy(raw / "train_images" / image_id, public / "test_images")
    
    # Copy label mapping file
    shutil.copy(raw / "label_num_to_disease_map.json", public / "label_num_to_disease_map.json")
    
    # Perform checks
    assert len(new_train) + len(new_test) == len(
        train_df
    ), "Expected new train and new test lengths to sum to original train length"
    assert len(sample_submission) == len(
        new_test
    ), "Expected sample submission length to be equal to new test length"
    
    assert len(new_train) == sum(
        1 for _ in (public / "train_images").iterdir()
    ), "Mismatch in number of expected train images copied"
    assert len(new_test) == sum(
        1 for _ in (public / "test_images").iterdir()
    ), "Mismatch in number of expected test images copied"
    
    assert new_train.columns.tolist() == [
        "image_id",
        "label",
    ], "Expected new train columns to be ['image_id', 'label']"
    assert new_test.columns.tolist() == [
        "image_id",
        "label",
    ], "Expected new test columns to be ['image_id', 'label']"
    assert sample_submission.columns.tolist() == [
        "image_id",
        "label",
    ], "Expected sample submission columns to be ['image_id', 'label']"
    
    assert set(new_train["image_id"]).isdisjoint(
        new_test["image_id"]
    ), "Expected train and test image IDs to be disjoint"

    # and cleanup
    shutil.rmtree(raw)
