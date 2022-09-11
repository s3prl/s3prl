import pytest
from dotenv import dotenv_values

from s3prl.dataio.corpus.quesst14 import quesst14_for_qbe


@pytest.mark.corpus
def test_quesst14_for_qbe():
    quesst_root = dotenv_values()["Quesst14"]
    all_data, valid_keys, test_keys, doc_keys = quesst14_for_qbe(quesst_root).slice(4)
    assert len(all_data) == 2714
    assert len(valid_keys) + len(test_keys) + len(doc_keys) == 2714
