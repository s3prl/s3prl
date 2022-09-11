from pathlib import Path

import pytest
from dotenv import dotenv_values

from s3prl import set_use_cache
from s3prl.dataio.corpus.voxceleb1sid import voxceleb1_for_utt_classification


@pytest.mark.corpus
@pytest.mark.parametrize("use_cache", [False, True])
def test_voxceleb1sid(use_cache):
    with set_use_cache(use_cache):
        config = dotenv_values()
        voxceleb1 = Path(config["VoxCeleb1"])
        if voxceleb1.is_dir():
            train_data, valid_data, test_data, stats = voxceleb1_for_utt_classification(
                voxceleb1
            ).split(3)
        else:
            raise ValueError("Please set the VoxCeleb1 path in .env")
