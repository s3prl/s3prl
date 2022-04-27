import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

from joblib import Parallel, delayed
from tqdm import tqdm

from s3prl import Object, Output, cache
from s3prl.util.loader import TorchaudioLoader
from s3prl.util.tokenizer import load_tokenizer

LIBRI_SPLITS = [
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
]

CHARACTER_VOCAB = list(" ABCDEFGHIJKLMNOPQRSTUVWXYZ'")
PHONEME_VOCAB = "SIL SPN AA0 AA1 AA2 AE0 AE1 AE2 AH0 AH1 AH2 AO0 AO1 AO2 AW0 AW1 AW2 AY0 AY1 AY2 B CH D DH EH0 EH1 EH2 ER0 ER1 ER2 EY0 EY1 EY2 F G HH IH0 IH1 IH2 IY0 IY1 IY2 JH K L M N NG OW0 OW1 OW2 OY0 OY1 OY2 P R S SH T TH UH0 UH1 UH2 UW0 UW1 UW2 V W Y Z ZH".split(
    " "
)

VOCAB_LIST = {
    "character": CHARACTER_VOCAB,
    "phoneme": PHONEME_VOCAB,
}


def read_text(file: str) -> str:
    src_file = "-".join(file.split("-")[:-1]) + ".trans.txt"
    idx = file.split("/")[-1].split(".")[0]

    with open(src_file, "r") as fp:
        for line in fp:
            if idx == line.split(" ")[0]:
                return line[:-1].split(" ", 1)[1]


class LibriSpeechCTCPreprocessor(Object):
    def __init__(
        self,
        dataset_root,
        splits: List[str] = LIBRI_SPLITS,
        vocab_type: str = "character",
        vocab_file: str = None,
        text_jobs: int = 8,
    ):
        super().__init__()

        dataset_root = Path(dataset_root).resolve()
        self.data_dict = self.collect_data(dataset_root, splits, text_jobs)

        if vocab_file is not None:
            self.tokenizer = load_tokenizer(vocab_type, vocab_file=vocab_file)
        else:
            self.tokenizer = load_tokenizer(
                vocab_type, vocab_list=VOCAB_LIST[vocab_type]
            )

    @staticmethod
    @cache()
    def collect_data(
        dataset_root, splits: List[str], n_jobs: int = 8
    ) -> Dict[str, List[str]]:

        data_dict = {}
        for split in splits:
            assert split in LIBRI_SPLITS, split

            wav_files = list(Path(dataset_root / split).rglob("*.flac"))
            text_list = Parallel(n_jobs=n_jobs)(
                delayed(read_text)(str(file)) for file in wav_files
            )
            data_dict[split] = {"path": wav_files, "label": text_list}

        return data_dict

    def merge_splits(self, splits: List[str]) -> Tuple[List[Path], List[str]]:
        source, label = [], []
        for split in splits:
            assert split in self.data_dict
            source += self.data_dict[split]["path"]
            label += self.data_dict[split]["label"]
        return source, label

    def train_data(self, splits: List[str] = ["train-clean-100"]) -> Output:
        source, label = self.merge_splits(splits)
        return Output(
            source=source,
            label=label,
            source_loader=TorchaudioLoader(),
            label_loader=self.tokenizer,
        )

    def valid_data(self, splits: List[str] = ["dev-clean"]) -> Output:
        source, label = self.merge_splits(splits)
        return Output(
            source=source,
            label=label,
            source_loader=TorchaudioLoader(),
            label_loader=self.tokenizer,
        )

    def test_data(self, splits: List[str] = ["test-clean"]) -> Output:
        source, label = self.merge_splits(splits)
        return Output(
            source=source,
            label=label,
            source_loader=TorchaudioLoader(),
            label_loader=self.tokenizer,
        )

    def statistics(self) -> Output:
        return Output(
            output_size=self.tokenizer.vocab_size, label_loader=self.tokenizer
        )
