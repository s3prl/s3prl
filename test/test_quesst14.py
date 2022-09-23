from pathlib import Path

import pytest
from dotenv import dotenv_values

from s3prl.dataio.corpus.quesst14 import Quesst14


@pytest.mark.corpus
def test_quesst14_for_qbe():
    def quesst14_for_qbe(dataset_root: str):
        corpus = Quesst14(dataset_root)

        def path_to_dict(path: str):
            return dict(
                wav_path=path,
            )

        return dict(
            all_data={
                Path(path).stem: path_to_dict(path)
                for path in (corpus.valid_queries + corpus.test_queries + corpus.docs)
            },
            valid_keys=[Path(path).stem for path in corpus.valid_queries],
            test_keys=[Path(path).stem for path in corpus.test_queries],
            doc_keys=[Path(path).stem for path in corpus.docs],
        )

    quesst_root = dotenv_values()["Quesst14"]
    all_data, valid_keys, test_keys, doc_keys = quesst14_for_qbe(quesst_root).values()
    assert len(all_data) == 2714
    assert len(valid_keys) + len(test_keys) + len(doc_keys) == 2714
