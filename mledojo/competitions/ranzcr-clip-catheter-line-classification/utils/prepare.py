import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

from mledojo.competitions.utils import read_csv

CLASSES = [
    "ETT - Abnormal",
    "ETT - Borderline",
    "ETT - Normal",
    "NGT - Abnormal",
    "NGT - Borderline",
    "NGT - Incompletely Imaged",
    "NGT - Normal",
    "CVC - Abnormal",
    "CVC - Borderline",
]


def prepare(raw: Path, public: Path, private: Path):
    # Create train, test from train split
    old_train = read_csv(raw / "train.csv")
    new_train, new_test = train_test_split(old_train, test_size=0.1, random_state=0)

    old_train_annotations = read_csv(raw / "train_annotations.csv")
    old_train_uids = old_train_annotations["StudyInstanceUID"]
    new_train_uids = new_train["StudyInstanceUID"]
    is_in_new_train = old_train_uids.isin(new_train_uids)

    new_train_annotations = old_train_annotations[is_in_new_train]

    (public / "train").mkdir(exist_ok=True)
    (public / "test").mkdir(exist_ok=True)

    for file_id in new_train["StudyInstanceUID"]:
        shutil.copyfile(
            src=raw / "train" / f"{file_id}.jpg",
            dst=public / "train" / f"{file_id}.jpg",
        )

    for file_id in new_test["StudyInstanceUID"]:
        shutil.copyfile(
            src=raw / "train" / f"{file_id}.jpg",
            dst=public / "test" / f"{file_id}.jpg",
        )

    assert len(list(public.glob("train/*.jpg"))) == len(
        new_train
    ), f"Expected {len(new_train)} files in public train, got {len(list(public.glob('train/*.jpg')))}"
    assert len(list(public.glob("test/*.jpg"))) == len(
        new_test
    ), f"Expected {len(new_test)} files in public test, got {len(list(public.glob('test/*.jpg')))}"

    # Create a sample submission file with random column values copied from test_answer
    submission_df = new_test.copy()
    random_col = CLASSES[0]  # Use first class column as source
    for col in CLASSES:
        submission_df[col] = submission_df[random_col]

    # Copy over files
    new_train.to_csv(public / "train.csv", index=False)
    new_train_annotations.to_csv(public / "train_annotations.csv", index=False)
    new_test.to_csv(private / "test_answer.csv", index=False)
    submission_df.to_csv(public / "sample_submission.csv", index=False)

    # clean up
    shutil.rmtree(raw)
