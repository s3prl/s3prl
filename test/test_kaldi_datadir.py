import pytest
from dotenv import dotenv_values

from s3prl.corpus.kaldi import kaldi_for_multiclass_tagging
from s3prl.dataset.multiclass_tagging import BuildMultiClassTagging


@pytest.mark.corpus
def test_kaldi_datadir_for_diar():
    kaldi_dir = dotenv_values()["Diarization"]
    train_data, valid_data, test_data = kaldi_for_multiclass_tagging(kaldi_dir).slice(3)
    test_dataset = BuildMultiClassTagging(feat_frame_shift=160)(test_data)
