import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

from mledojo.competitions.utils import df_to_one_hot, read_csv

_dogs_str = """
affenpinscher
afghan_hound
african_hunting_dog
airedale
american_staffordshire_terrier
appenzeller
australian_terrier
basenji
basset
beagle
bedlington_terrier
bernese_mountain_dog
black-and-tan_coonhound
blenheim_spaniel
bloodhound
bluetick
border_collie
border_terrier
borzoi
boston_bull
bouvier_des_flandres
boxer
brabancon_griffon
briard
brittany_spaniel
bull_mastiff
cairn
cardigan
chesapeake_bay_retriever
chihuahua
chow
clumber
cocker_spaniel
collie
curly-coated_retriever
dandie_dinmont
dhole
dingo
doberman
english_foxhound
english_setter
english_springer
entlebucher
eskimo_dog
flat-coated_retriever
french_bulldog
german_shepherd
german_short-haired_pointer
giant_schnauzer
golden_retriever
gordon_setter
great_dane
great_pyrenees
greater_swiss_mountain_dog
groenendael
ibizan_hound
irish_setter
irish_terrier
irish_water_spaniel
irish_wolfhound
italian_greyhound
japanese_spaniel
keeshond
kelpie
kerry_blue_terrier
komondor
kuvasz
labrador_retriever
lakeland_terrier
leonberg
lhasa
malamute
malinois
maltese_dog
mexican_hairless
miniature_pinscher
miniature_poodle
miniature_schnauzer
newfoundland
norfolk_terrier
norwegian_elkhound
norwich_terrier
old_english_sheepdog
otterhound
papillon
pekinese
pembroke
pomeranian
pug
redbone
rhodesian_ridgeback
rottweiler
saint_bernard
saluki
samoyed
schipperke
scotch_terrier
scottish_deerhound
sealyham_terrier
shetland_sheepdog
shih-tzu
siberian_husky
silky_terrier
soft-coated_wheaten_terrier
staffordshire_bullterrier
standard_poodle
standard_schnauzer
sussex_spaniel
tibetan_mastiff
tibetan_terrier
toy_poodle
toy_terrier
vizsla
walker_hound
weimaraner
welsh_springer_spaniel
west_highland_white_terrier
whippet
wire-haired_fox_terrier
yorkshire_terrier
"""
DOGS_LIST = _dogs_str.strip().split("\n")
assert len(DOGS_LIST) == 120


def to_one_hot(df):
    return df_to_one_hot(df, id_column="id", target_column="breed", classes=DOGS_LIST)


def prepare(raw: Path, public: Path, private: Path):

    # Create train, test from train split
    old_train = read_csv(raw / "labels.csv")
    new_train, new_test = train_test_split(old_train, test_size=0.1, random_state=0)
    # one hot the private test because we will one-hot the submission, as per kaggle.com
    new_test = to_one_hot(new_test)

    # Copy over files
    new_train.to_csv(public / "labels.csv", index=False)
    new_test.to_csv(private / "test_answer.csv", index=False)

    (public / "train").mkdir(exist_ok=True)

    for file_id in new_train["id"]:
        shutil.copyfile(
            src=raw / "train" / f"{file_id}.jpg",
            dst=public / "train" / f"{file_id}.jpg",
        )

    (public / "test").mkdir(exist_ok=True)

    for file_id in new_test["id"]:
        shutil.copyfile(
            src=raw / "train" / f"{file_id}.jpg",
            dst=public / "test" / f"{file_id}.jpg",
        )

    # Check integrity of the files copied
    assert len(list(public.glob("train/*.jpg"))) == len(new_train)
    assert len(list(public.glob("test/*.jpg"))) == len(new_test)

    # Create a sample submission file
    submission_df = new_test.copy()
    for col in submission_df.columns[1:]:
        submission_df[col] = submission_df[col].astype("float64")
    submission_df.iloc[:, 1:] = 1 / 120
    submission_df.to_csv(public / "sample_submission.csv", index=False)

    assert submission_df.shape == (len(new_test), 121)  # 1 id column + 120 breeds

    # and cleanup
    shutil.rmtree(raw)
