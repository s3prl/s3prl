"""
Parse the Audio SNIPS corpus

Authors:
  * Heng-Jui Chang 2022
"""

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

from tqdm import trange

from .base import Corpus

__all__ = [
    "SNIPS",
]


class SNIPS(Corpus):
    def __init__(
        self,
        dataset_root: str,
        train_speakers: List[str],
        valid_speakers: List[str],
        test_speakers: List[str],
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.train_speakers = train_speakers
        self.valid_speakers = valid_speakers
        self.test_speakers = test_speakers

        self.data_dict = self._collect_data(
            self.dataset_root, train_speakers, valid_speakers, test_speakers
        )
        self.train = self._data_to_dict(self.data_dict, ["train"])
        self.valid = self._data_to_dict(self.data_dict, ["valid"])
        self.test = self._data_to_dict(self.data_dict, ["test"])

        self._data = OrderedDict()
        self._data.update(self.train)
        self._data.update(self.valid)
        self._data.update(self.test)

    @property
    def all_data(self):
        return self._data

    @property
    def data_split_ids(self):
        return (
            list(self.train.keys()),
            list(self.valid.keys()),
            list(self.test.keys()),
        )

    @staticmethod
    def _collect_data(
        dataset_root: str,
        train_speakers: List[str],
        valid_speakers: List[str],
        test_speakers: List[str],
    ) -> Dict[str, Dict[str, Any]]:

        # Load transcription
        transcripts_file = open(dataset_root / "all.iob.snips.txt").readlines()
        transcripts = {}
        for line in transcripts_file:
            line = line.strip().split(" ")
            index = line[0]  # {speaker}-snips-{split}-{index}
            sent = " ".join(line[1:])
            transcripts[index] = sent

        # List wave files
        data_dict = {}
        for split, speaker_list in [
            ("train", train_speakers),
            ("valid", valid_speakers),
            ("test", test_speakers),
        ]:
            wav_list = list((dataset_root / split).rglob("*.wav"))
            new_wav_list, name_list, spkr_list = [], [], []
            uf = 0
            for i in trange(len(wav_list), desc="checking files"):
                uid = wav_list[i].stem
                if uid in transcripts:
                    spkr = uid.split("-")[0]
                    if spkr in speaker_list:
                        new_wav_list.append(str(wav_list[i]))
                        name_list.append(uid)
                        spkr_list.append(spkr)
                else:
                    logging.info(wav_list[i], "Not Found")
                    uf += 1

            logging.info("%d wav file with label not found in text file!" % uf)
            wav_list = new_wav_list
            logging.info(
                f"loaded audio from {len(speaker_list)} speakers {str(speaker_list)} with {len(wav_list)} examples."
            )
            assert len(wav_list) > 0, "No data found @ {}".format(dataset_root / split)

            text_list = [transcripts[name] for name in name_list]

            wav_list, name_list, text_list, spkr_list = zip(
                *[
                    (wav, name, text, spkr)
                    for (wav, name, text, spkr) in sorted(
                        zip(wav_list, name_list, text_list, spkr_list),
                        key=lambda x: x[1],
                    )
                ]
            )

            data_dict[split] = {
                "name_list": name_list,
                "wav_list": wav_list,
                "text_list": text_list,
                "spkr_list": spkr_list,
            }

        return data_dict

    @staticmethod
    def _data_to_dict(
        data_dict: Dict[str, Dict[str, List[Any]]], splits: List[str]
    ) -> dict:
        data = dict(
            {
                name: {
                    "wav_path": data_dict[split]["wav_list"][i],
                    "transcription": " ".join(
                        data_dict[split]["text_list"][i]
                        .split("\t")[0]
                        .strip()
                        .split(" ")[1:-1]
                    ),
                    "iob": " ".join(
                        data_dict[split]["text_list"][i]
                        .split("\t")[1]
                        .strip()
                        .split(" ")[1:-1]
                    ),
                    "intent": data_dict[split]["text_list"][i]
                    .split("\t")[1]
                    .strip()
                    .split(" ")[-1],
                    "speaker": data_dict[split]["spkr_list"][i],
                    "corpus_split": split,
                }
                for split in splits
                for i, name in enumerate(data_dict[split]["name_list"])
            }
        )
        return data
