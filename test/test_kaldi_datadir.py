from pathlib import Path

import pytest
from dotenv import dotenv_values

from s3prl import Workspace
from s3prl.base import fileio
from s3prl.corpus.kaldi import kaldi_for_multiclass_tagging
from s3prl.dataset.multiclass_tagging import BuildMultiClassTagging


@pytest.mark.corpus
def test_kaldi_datadir_for_diar():
    kaldi_dir = dotenv_values()["Diarization"]
    train_data, valid_data, test_data = kaldi_for_multiclass_tagging(kaldi_dir).slice(3)

    ws = Workspace()
    fileio.save(
        (ws / "test").resolve(),
        fileio.as_type(
            {reco: value["segments"] for reco, value in test_data.items()}, "rttm"
        ),
    )

    ori_rttm = fileio.load(Path(kaldi_dir) / "test" / "rttm", "rttm")
    new_rttm = fileio.load((ws / "test.rttm").resolve(), "rttm")

    assert ori_rttm == new_rttm

    train_dataset = BuildMultiClassTagging()(train_data)
    first_item = train_dataset[0]
