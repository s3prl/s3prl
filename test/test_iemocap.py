import pytest
from dotenv import dotenv_values

from s3prl.corpus.iemocap import IEMOCAP, iemocap_for_superb


@pytest.mark.corpus
def test_iemocap():
    config = dotenv_values()
    dataset_root = config["IEMOCAP"]
    dataset = IEMOCAP(dataset_root)
    data = dataset.all_data

    train_data, valid_data, test_data, stats = iemocap_for_superb(dataset_root).split(3)
