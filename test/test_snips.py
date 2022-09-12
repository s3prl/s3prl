import pytest
from dotenv import dotenv_values

from s3prl.dataio.corpus.snips import SNIPS


@pytest.mark.corpus
def test_snips():
    config = dotenv_values()
    dataset_root = config["SNIPS"]
    dataset = SNIPS(
        dataset_root,
        [
            "Ivy",
            "Joanna",
            "Joey",
            "Justin",
            "Kendra",
            "Kimberly",
            "Matthew",
            "Salli",
        ],
        ["Aditi", "Amy", "Geraint", "Nicole"],
        ["Brian", "Emma", "Raveena", "Russell"],
    )
    train_data, valid_data, test_data = dataset.data_split

    assert len(train_data) == 104672
    assert len(valid_data) == 2800
    assert len(test_data) == 2800
