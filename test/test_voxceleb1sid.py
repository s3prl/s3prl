from pathlib import Path

import pytest
from dotenv import dotenv_values

from s3prl import set_use_cache
from s3prl.corpus.voxceleb1sid import VoxCeleb1SIDForUtteranceClassification


@pytest.mark.parametrize("use_cache", [False, True])
def test_voxceleb1sid(use_cache):
    with set_use_cache(use_cache):
        config = dotenv_values()
        voxceleb1 = Path(config["VoxCeleb1"])
        if voxceleb1.is_dir():
            corpus = VoxCeleb1SIDForUtteranceClassification(voxceleb1)
            train_data, valid_data, test_data, stats = corpus().split(3)
        else:
            raise ValueError("Please set the VoxCeleb1 path in .env")
