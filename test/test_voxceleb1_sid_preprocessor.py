from pathlib import Path

import pytest
from dotenv import dotenv_values

from s3prl import set_use_cache
from s3prl.preprocessor import VoxCeleb1SIDPreprocessor


@pytest.mark.parametrize("use_cache", [False, True])
def test_voxceleb1_sid_preprocessor(use_cache):
    with set_use_cache(use_cache):
        config = dotenv_values()
        voxceleb1 = Path(config["VoxCeleb1"])
        if voxceleb1.is_dir():
            preprocessor = VoxCeleb1SIDPreprocessor(voxceleb1)
            preprocessor.train_data()
            preprocessor.valid_data()
            preprocessor.test_data()
            preprocessor.statistics()
        else:
            raise ValueError("Please set the VoxCeleb1 path in .env")
