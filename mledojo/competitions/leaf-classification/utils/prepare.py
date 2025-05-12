import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

from mledojo.competitions.utils import df_to_one_hot, extract, read_csv

CLASSES = [
    "Acer_Capillipes",
    "Acer_Circinatum",
    "Acer_Mono",
    "Acer_Opalus",
    "Acer_Palmatum",
    "Acer_Pictum",
    "Acer_Platanoids",
    "Acer_Rubrum",
    "Acer_Rufinerve",
    "Acer_Saccharinum",
    "Alnus_Cordata",
    "Alnus_Maximowiczii",
    "Alnus_Rubra",
    "Alnus_Sieboldiana",
    "Alnus_Viridis",
    "Arundinaria_Simonii",
    "Betula_Austrosinensis",
    "Betula_Pendula",
    "Callicarpa_Bodinieri",
    "Castanea_Sativa",
    "Celtis_Koraiensis",
    "Cercis_Siliquastrum",
    "Cornus_Chinensis",
    "Cornus_Controversa",
    "Cornus_Macrophylla",
    "Cotinus_Coggygria",
    "Crataegus_Monogyna",
    "Cytisus_Battandieri",
    "Eucalyptus_Glaucescens",
    "Eucalyptus_Neglecta",
    "Eucalyptus_Urnigera",
    "Fagus_Sylvatica",
    "Ginkgo_Biloba",
    "Ilex_Aquifolium",
    "Ilex_Cornuta",
    "Liquidambar_Styraciflua",
    "Liriodendron_Tulipifera",
    "Lithocarpus_Cleistocarpus",
    "Lithocarpus_Edulis",
    "Magnolia_Heptapeta",
    "Magnolia_Salicifolia",
    "Morus_Nigra",
    "Olea_Europaea",
    "Phildelphus",
    "Populus_Adenopoda",
    "Populus_Grandidentata",
    "Populus_Nigra",
    "Prunus_Avium",
    "Prunus_X_Shmittii",
    "Pterocarya_Stenoptera",
    "Quercus_Afares",
    "Quercus_Agrifolia",
    "Quercus_Alnifolia",
    "Quercus_Brantii",
    "Quercus_Canariensis",
    "Quercus_Castaneifolia",
    "Quercus_Cerris",
    "Quercus_Chrysolepis",
    "Quercus_Coccifera",
    "Quercus_Coccinea",
    "Quercus_Crassifolia",
    "Quercus_Crassipes",
    "Quercus_Dolicholepis",
    "Quercus_Ellipsoidalis",
    "Quercus_Greggii",
    "Quercus_Hartwissiana",
    "Quercus_Ilex",
    "Quercus_Imbricaria",
    "Quercus_Infectoria_sub",
    "Quercus_Kewensis",
    "Quercus_Nigra",
    "Quercus_Palustris",
    "Quercus_Phellos",
    "Quercus_Phillyraeoides",
    "Quercus_Pontica",
    "Quercus_Pubescens",
    "Quercus_Pyrenaica",
    "Quercus_Rhysophylla",
    "Quercus_Rubra",
    "Quercus_Semecarpifolia",
    "Quercus_Shumardii",
    "Quercus_Suber",
    "Quercus_Texana",
    "Quercus_Trojana",
    "Quercus_Variabilis",
    "Quercus_Vulcanica",
    "Quercus_x_Hispanica",
    "Quercus_x_Turneri",
    "Rhododendron_x_Russellianum",
    "Salix_Fragilis",
    "Salix_Intergra",
    "Sorbus_Aria",
    "Tilia_Oliveri",
    "Tilia_Platyphyllos",
    "Tilia_Tomentosa",
    "Ulmus_Bergmanniana",
    "Viburnum_Tinus",
    "Viburnum_x_Rhytidophylloides",
    "Zelkova_Serrata",
]


def prepare(raw: Path, public: Path, private: Path):
    """
    Splits the data in raw into public and private datasets with appropriate test/train splits.
    """
    # extract only what we need
    extract(raw / "train.csv.zip", raw)
    extract(raw / "images.zip", raw)

    # Create train, test from train split
    old_train = read_csv(raw / "train.csv")
    new_train, new_test = train_test_split(old_train, test_size=0.1, random_state=0)
    new_test_without_labels = new_test.drop(columns=["species"])

    # match the format of the sample submission
    new_test = new_test[["id", "species"]]
    new_test = df_to_one_hot(new_test, "id", "species", classes=CLASSES)

    (public / "images").mkdir(exist_ok=True)
    (private / "images").mkdir(exist_ok=True)

    for file_id in new_train["id"]:
        shutil.copyfile(
            src=raw / "images" / f"{file_id}.jpg",
            dst=public / "images" / f"{file_id}.jpg",
        )

    for file_id in new_test_without_labels["id"]:
        shutil.copyfile(
            src=raw / "images" / f"{file_id}.jpg",
            dst=public / "images" / f"{file_id}.jpg",
        )

    # Check integrity of the files copied
    assert len(new_test_without_labels) == len(
        new_test
    ), "Public and Private tests should have equal length"
    assert len(list(public.glob("images/*.jpg"))) == len(new_train) + len(
        new_test_without_labels
    ), "Public images should have the same number of images as the sum of train and test"

    # Create a sample submission file
    submission_df = new_test.copy()
    submission_df[CLASSES] = 1 / len(CLASSES)

    # Copy over files
    new_train.to_csv(public / "train.csv", index=False)
    new_test.to_csv(private / "test_answer.csv", index=False)
    new_test_without_labels.to_csv(public / "test.csv", index=False)
    submission_df.to_csv(public / "sample_submission.csv", index=False)
    
    if (private / "images").exists():
        shutil.rmtree(private / "images")
    # and cleanup
    shutil.rmtree(raw)
