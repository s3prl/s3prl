import pytest
from dotenv import dotenv_values

from s3prl.dataio.corpus.snips import snips_for_speech2text


@pytest.mark.corpus
def test_snips():
    config = dotenv_values()
    dataset_root = config["SNIPS"]
    dataset = snips_for_speech2text(dataset_root)

    assert len(dataset.train_data) == 104672
    assert len(dataset.valid_data) == 2800
    assert len(dataset.test_data) == 2800
