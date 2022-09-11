import logging
import os
import tempfile

import pytest
from dotenv import dotenv_values

from s3prl.dataio.corpus.librispeech import librispeech_for_speech2text
from s3prl.dataset.base import AugmentedDynamicItemDataset
from s3prl.dataset.speech2text_pipe import Speech2TextPipe


@pytest.mark.corpus
def test_speech2text_pipe():
    config = dotenv_values()
    train_data, valid_data, test_data, corpus_stats = librispeech_for_speech2text(
        config["LibriSpeech"]
    ).split(3)

    dataset = AugmentedDynamicItemDataset(train_data)
    dataset_char = Speech2TextPipe(generate_tokenizer=True, vocab_type="character")(
        dataset
    )

    assert dataset_char.get_tool("output_size") == 31

    with tempfile.TemporaryDirectory() as directory:
        logging.info(directory)
        output_file = os.path.join(directory, "subword")
        dataset = AugmentedDynamicItemDataset(train_data)
        dataset_subword = Speech2TextPipe(
            generate_tokenizer=True,
            vocab_type="subword",
            vocab_args={"vocab_size": 4000, "output_file": output_file},
        )(dataset)

        assert dataset_subword.get_tool("output_size") == 4000
