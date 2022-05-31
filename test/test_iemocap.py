import pytest
from dotenv import dotenv_values

from s3prl.corpus.iemocap import IEMOCAP, IEMOCAPForSUPERB


@pytest.mark.corpus
def test_iemocap():
    config = dotenv_values()
    dataset_root = config["IEMOCAP"]
    dataset = IEMOCAP(dataset_root)
    data = dataset.all_data

    corpus = IEMOCAPForSUPERB(dataset_root)
    train_data, valid_data, test_data, stats = corpus().split(3)
