import pytest
from dotenv import dotenv_values

from s3prl.corpus.iemocap import IEMOCAP, iemocap_for_superb
from s3prl.util.download import _urls_to_filepaths
from s3prl.base import fileio


@pytest.mark.corpus
def test_iemocap():
    config = dotenv_values()
    dataset_root = config["IEMOCAP"]

    train_data, valid_data, test_data, stats = iemocap_for_superb(dataset_root).split(3)
    train_ids, valid_ids, test_ids = _urls_to_filepaths(
        "https://huggingface.co/datasets/s3prl/iemocap_split/raw/main/Session1/train.txt",
        "https://huggingface.co/datasets/s3prl/iemocap_split/raw/main/Session1/valid.txt",
        "https://huggingface.co/datasets/s3prl/iemocap_split/raw/main/Session1/test.txt",
    )
    train_ids = [line.strip() for line in open(train_ids).readlines()]
    valid_ids = [line.strip() for line in open(valid_ids).readlines()]
    test_ids = [line.strip() for line in open(test_ids).readlines()]

    assert sorted(train_data.keys()) == sorted(train_ids)
    assert sorted(valid_data.keys()) == sorted(valid_ids)
    assert sorted(test_data.keys()) == sorted(test_ids)
