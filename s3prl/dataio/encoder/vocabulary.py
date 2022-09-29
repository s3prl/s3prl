"""
Create vocabulary (train tokenizer)

Authors:
  * Heng-Jui Chang 2022
"""

import logging
import os
import tempfile
from collections import Counter
from typing import List, Union

logger = logging.getLogger(__name__)

__all__ = ["generate_basic_vocab", "generate_subword_vocab", "generate_vocab"]


def generate_basic_vocab(
    mode: str,
    text_list: List[str],
    vocab_size: int = -1,
    coverage: float = 1.0,
    sort_vocab: bool = True,
) -> List[str]:
    """Generates basic vocabularies, including character and word-based vocabularies.

    Args:
        mode (str): Vocabulary type (character or word).
        text_list (List[str]): List of text data.
        vocab_size (int, optional):
            Vocabulary size, if not specified, vocab_size would be `coverage * actual vocab size`. Defaults to -1.
        coverage (float, optional): Vocabulary coverage. Defaults to 1.0.
        sort_vocab (bool, optional): Sort vocabularies alphabetically. Defaults to True.

    Returns:
        List[str]: A list of vocabularies.
    """

    assert mode in {"character", "word"}, mode
    assert vocab_size == -1 or vocab_size > 0, vocab_size
    assert coverage > 0.0 and coverage <= 1.0, coverage

    logger.info(
        f"Generating vocab (type = {mode}, coverage = {coverage}) from {len(text_list)} sentences."
    )

    counter = Counter()

    for text in text_list:
        if mode == "character":
            counter.update(text)
        if mode == "word":
            counter.update(text.split())

    if vocab_size < 0:
        vocab_size = int(len(counter) * coverage)
    else:
        vocab_size = min(vocab_size, len(counter))

    if vocab_size < len(counter):
        vocab_list = sorted(counter.keys(), key=lambda k: counter[k], reverse=True)
        vocab_list = vocab_list[:vocab_size]
    else:
        vocab_list = list(counter.keys())

    if sort_vocab:
        vocab_list = sorted(vocab_list)

    logger.info(f"Generated {vocab_size} {mode} vocabularies.")

    return vocab_list


def generate_subword_vocab(
    text_list: List[str] = None,
    text_file: str = None,
    output_file: str = None,
    vocab_size: int = 1000,
    character_coverage: float = 1.0,
) -> str:
    """Generates subword vocabularies based on `sentencepiece`.

    Args:
        text_list (List[str], optional): List of text data. Defaults to None.
        text_file (str, optional): Path to text data. Defaults to None.
        output_file (str, optional): Path to save trained subword vocabularies. Defaults to "".
        vocab_size (int, optional): Vocabulary size. Defaults to 8000.
        character_coverage (float, optional): Coverage of characters in text data. Defaults to 1.0.

    Raises:
        ImportError: If `sentencepiece` is not installed.

    Returns:
        str: Path to `${output_file}.model`.
    """

    try:
        import sentencepiece as splib
    except ImportError:
        raise ImportError(
            "`sentencepiece` cannot be imported, please run `pip install sentencepiece` first"
        )

    assert output_file is not None
    output_file = str(output_file)
    assert vocab_size > 0, vocab_size

    cmd = (
        "--input={} --model_prefix={} --model_type=unigram "
        "--vocab_size={} --character_coverage={} "
        "--pad_id=0 --eos_id=1 --unk_id=2 --bos_id=-1 "
        "--eos_piece=<eos> --remove_extra_whitespaces=true "
    )

    if text_list is not None:
        assert isinstance(text_list, list)
        assert isinstance(text_list[0], str)

        logger.info(
            f"Generating vocab (type = subword, coverage = {character_coverage}) from {len(text_list)} sentences."
        )

        with tempfile.TemporaryDirectory() as directory:
            input_file = os.path.join(directory, "text.txt")
            with open(input_file, "w") as fp:
                for text in text_list:
                    fp.write(text + "\n")

            cmd = cmd.format(
                input_file,
                output_file,
                vocab_size,
                character_coverage,
            )
            splib.SentencePieceTrainer.Train(cmd)

    if text_file is not None:
        logger.info(
            f"Generating vocab (type = subword, coverage = {character_coverage}) from {text_file}"
        )

        cmd = cmd.format(
            text_file,
            output_file,
            vocab_size,
            character_coverage,
        )
        splib.SentencePieceTrainer.Train(cmd)

    return output_file + ".model"


def generate_vocab(
    mode: str,
    text_list: List[str] = None,
    text_file: str = None,
    read_lines: int = 10000000,
    **vocab_args,
) -> Union[List[str], str]:
    """Generates vocabularies given text data.

    Args:
        mode (str): Vocabulary type
        text_list (List[str], optional): List of text data. Defaults to None.
        text_file (str, optional): Path to text data. Defaults to None.
        read_lines (int, optional): Maximum lines to read from `text_file`. Defaults to 10000000.
        vocab_args:
            if :code:`mode != subword`, arguments for :obj:`generate_basic_vocab`
            if :code:`mode == subword`, arguments for :obj:`generate_subword_vocab`

    Returns:
        Union[List[str], str]: A list of vocabularies or a path to `.vocab` file.
    """

    if text_list is None and mode in {"character", "word", "phoneme"}:
        assert isinstance(text_file, str)
        with open(text_file, "r", encoding="UTF-8") as fp:
            text_list = [
                line.strip("\r\n ") for i, line in enumerate(fp) if i < read_lines
            ]

    if mode == "character":
        return generate_basic_vocab("character", text_list, **vocab_args)
    if mode in {"word", "phoneme"}:
        return generate_basic_vocab("word", text_list, **vocab_args)
    if mode == "subword":
        return generate_subword_vocab(
            text_list=text_list, text_file=text_file, **vocab_args
        )
    else:
        raise ValueError(f"Unsupported mode (vocabulary type): {mode}")
