import logging
from collections import defaultdict
from typing import Dict, List, Tuple


def parse_lexicon(line: str) -> Tuple[str, List[str]]:
    line.replace("\t", " ")
    word, *phonemes = line.split()
    return word, phonemes


def read_lexicon_files(file_list: List[str]) -> Dict[str, List[str]]:
    w2p_dict = defaultdict(list)
    for file in file_list:
        with open(file, "r") as fp:
            lines = [line.strip() for line in fp]
            for line in lines:
                word, phonemes = parse_lexicon(line)
                w2p_dict[word].append(phonemes)

    w2p = {}
    for word, phonemes_all in w2p_dict.items():
        if len(phonemes_all) > 1:
            logging.info(f"{len(phonemes_all)} phoneme sequences found for {word}.")
            for i, phonemes in enumerate(phonemes_all):
                logging.info(f"{i}. {phonemes}")
        w2p[word] = phonemes_all[0]
    logging.info(f"Taking the first phoneme sequences for a deterministic behavior.")

    return w2p


class G2P:
    def __init__(self):
        # TODO: Download lexicon

        file_list = [
            "s3prl/downstream/ctc/lexicon/librispeech-lexicon-200k-g2p.txt",
            "s3prl/downstream/ctc/lexicon/librispeech-lexicon-allothers-g2p.txt",
        ]

        self.word2phone = read_lexicon_files(file_list)

    def __call__(self, text: str) -> str:
        word_list = text.split(" ")
        phonemes = []
        for word in word_list:
            phonemes += self.word2phone.get(word, ["<UNK>"])
        return " ".join(phonemes)
