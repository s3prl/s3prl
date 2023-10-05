from pathlib import Path

import pytest
from dotenv import dotenv_values

from s3prl.dataio.corpus.voxceleb1sid import VoxCeleb1SID


@pytest.mark.corpus
@pytest.mark.parametrize("use_cache", [False, True])
def test_voxceleb1sid(use_cache):
    config = dotenv_values()
    voxceleb1 = Path(config["VoxCeleb1"])
    if voxceleb1.is_dir():
        train_data, valid_data, test_data = VoxCeleb1SID(voxceleb1).data_split
    else:
        raise ValueError("Please set the VoxCeleb1 path in .env")
