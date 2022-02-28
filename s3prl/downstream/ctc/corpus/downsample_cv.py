import argparse
import csv
from os.path import join
from pathlib import Path
from tqdm import tqdm

import torch
import torchaudio
import numpy as np
from librosa import resample


def read_processed_tsv(path):
    with open(path, "r") as fp:
        rows = csv.reader(fp, delimiter="\t")
        file_list = []
        for i, row in enumerate(rows):
            if i == 0:
                continue
            file_list.append(row[0][:-3] + "mp3")
        return file_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, help="Directory of the dataset.")
    parser.add_argument("--tsv", type=str, help="Path to processed tsv file.")
    args = parser.parse_args()

    file_list = read_processed_tsv(args.tsv)

    for file in tqdm(file_list):
        file = str(file)
        file = join(args.root, file)
        wav, sample_rate = torchaudio.load(file)
        wav = resample(
            wav.squeeze(0).numpy(), sample_rate, 16000, res_type="kaiser_best"
        )
        wav = torch.FloatTensor(wav).unsqueeze(0)
        new_file = file[:-3] + "wav"
        torchaudio.save(new_file, wav, 16000)


if __name__ == "__main__":
    main()
