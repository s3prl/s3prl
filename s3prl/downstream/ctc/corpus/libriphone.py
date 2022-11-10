import logging

import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def parse_lexicon(line, tokenizer):
    line.replace("\t", " ")
    word, *phonemes = line.split()
    for p in phonemes:
        assert p in tokenizer._vocab2idx.keys()
    return word, phonemes


def phonemize(transcription, word2phonemes, tokenizer):
    phonemes = []
    for word in transcription.split():
        phonemes += word2phonemes[word]
    return tokenizer.encode(" ".join(phonemes))


class LibriPhoneDataset(Dataset):
    """
    Args:
        split_csv (str): path for train.csv / valid.csv / test.csv
    """

    def __init__(
        self,
        split_csv,
        tokenizer,
        bucket_size,
        lexicon,
        csv_dir,
        ascending=False,
        **kwargs,
    ):
        self.bucket_size = bucket_size

        # create word -> phonemes mapping
        word2phonemes_all = defaultdict(list)
        for lexicon_file in lexicon:
            with open(lexicon_file, "r") as file:
                lines = [line.strip() for line in file.readlines()]
                for line in lines:
                    word, phonemes = parse_lexicon(line, tokenizer)
                    word2phonemes_all[word].append(phonemes)

        # check mapping number of each word
        word2phonemes = {}
        for word, phonemes_all in word2phonemes_all.items():
            if len(phonemes_all) > 1:
                logger.info(
                    f"{len(phonemes_all)} of phoneme sequences found for {word}"
                )
                for idx, phonemes in enumerate(phonemes_all):
                    logger.info(f"{idx}. {phonemes}")
            word2phonemes[word] = phonemes_all[0]
        logger.info(f"Taking the first phoneme sequences for a deterministic behavior")

        df = pd.read_csv(Path(csv_dir) / split_csv)
        file_list = df["wav_path"].tolist()
        transcription = df["transcription"].tolist()

        text = []
        for trans in tqdm(transcription, desc="word -> phonemes"):
            text.append(phonemize(trans, word2phonemes, tokenizer))

        self.file_list, self.text = zip(
            *[
                (f_name, txt)
                for f_name, txt in sorted(
                    zip(file_list, text), reverse=not ascending, key=lambda x: len(x[1])
                )
            ]
        )

    def __getitem__(self, index):
        if self.bucket_size > 1:
            index = min(len(self.file_list) - self.bucket_size, index)
            return [
                (f_path, txt)
                for f_path, txt in zip(
                    self.file_list[index : index + self.bucket_size],
                    self.text[index : index + self.bucket_size],
                )
            ]
        else:
            return self.file_list[index], self.text[index]

    def __len__(self):
        return len(self.file_list)
