import logging
import os
import tempfile

import pytest
from dotenv import dotenv_values

from s3prl.dataio.corpus.librispeech import LibriSpeech
from s3prl.dataio.encoder.tokenizer import load_tokenizer
from s3prl.dataio.encoder.vocabulary import generate_vocab

SAMPLE = "GOOD MORNING MY FRIEND"


def is_same_vocab(vocabs_1, vocabs_2):
    if len(vocabs_1) != len(vocabs_2):
        return False

    for v1, v2 in zip(vocabs_1, vocabs_2):
        if v1 != v2:
            return False

    return True


@pytest.mark.corpus
def test_vocabulary():
    config = dotenv_values()
    corpus = LibriSpeech(config["LibriSpeech"])
    text_list = corpus.data_dict["train-clean-100"]["text_list"]

    with tempfile.TemporaryDirectory() as directory:
        logging.info(directory)
        text_file = os.path.join(directory, "text.txt")

        with open(text_file, "w") as fp:
            for text in text_list:
                fp.write(text + "\n")

        # Character
        char_vocabs_1 = generate_vocab("character", text_list=text_list)
        char_vocabs_2 = generate_vocab("character", text_file=text_file)

        assert isinstance(char_vocabs_1, list)
        assert isinstance(char_vocabs_2, list)
        assert is_same_vocab(char_vocabs_1, char_vocabs_2)

        char_tokenizer = load_tokenizer("character", vocab_list=char_vocabs_1)
        assert char_tokenizer.decode(char_tokenizer.encode(SAMPLE)) == SAMPLE

        # Word
        word_vocabs_1 = generate_vocab("word", text_list=text_list, vocab_size=5000)
        word_vocabs_2 = generate_vocab("word", text_file=text_file, vocab_size=5000)

        assert isinstance(word_vocabs_1, list)
        assert isinstance(word_vocabs_2, list)
        assert is_same_vocab(word_vocabs_1, word_vocabs_2)

        word_tokenizer = load_tokenizer("word", vocab_list=word_vocabs_1)
        assert word_tokenizer.decode(word_tokenizer.encode(SAMPLE)) == SAMPLE

        # Subword
        vocab_file_1 = os.path.join(directory, "subword_1")
        vocab_file_2 = os.path.join(directory, "subword_2")

        subword_vocabs_1 = generate_vocab(
            "subword", text_list=text_list, vocab_size=500, output_file=vocab_file_1
        )
        subword_vocabs_2 = generate_vocab(
            "subword", text_file=text_file, vocab_size=500, output_file=vocab_file_2
        )

        subword_tokenizer_1 = load_tokenizer(
            "subword", vocab_file=vocab_file_1 + ".model"
        )
        subword_tokenizer_2 = load_tokenizer(
            "subword", vocab_file=vocab_file_2 + ".model"
        )
        assert subword_tokenizer_1.decode(subword_tokenizer_1.encode(SAMPLE)) == SAMPLE
        assert subword_tokenizer_2.decode(subword_tokenizer_2.encode(SAMPLE)) == SAMPLE
        assert subword_tokenizer_1.encode(SAMPLE) == subword_tokenizer_2.encode(SAMPLE)
