import csv
from tqdm import tqdm
from pathlib import Path
from os.path import join, getsize
from torch.utils.data import Dataset
import argparse
import unicodedata
from string import punctuation
import re
import os
from os.path import join
from mutagen.mp3 import MP3


import torch
import torchaudio
import numpy as np
from librosa import resample


rm_punctuation = punctuation.replace("'", "")
rm_punctuation += r"""「」、⋯。《》丶–—―‘“”『』〜ー・☭«»،؛؟‐−¡¿ː ́·× ̃ ̌─〈〉"""
translator = str.maketrans(
    rm_punctuation + "’" + "ㄧ", " " * len(rm_punctuation) + "'" + "一"
)
spanish_exception = set({"し", "ら", "ゴ", "ミ", "ム", "ラ", "消", "箱", "良"})
zhcn_exception = set({"Μ", "И", "Т", "オ", "カ", "ド", "ヤ", "䴕"})


def normalize(sent, language):
    sent = unicodedata.normalize("NFKC", sent).upper()
    sent = sent.translate(translator)
    sent = re.sub(" +", " ", sent)
    if language in ["zh-TW", "zh-CN", "ja"]:
        sent = sent.replace(" ", "")
    if language in ["zh-TW", "zh-CN", "ja", "ar", "ru"]:
        if any([(c.encode("UTF-8").isalpha() or c == "'") for c in list(sent)]):
            return ""
    if language == "zh-CN":
        if len(zhcn_exception.intersection(set(list(sent)))) > 0:
            return ""
    if language == "es":
        if len(spanish_exception.intersection(set(list(sent)))) > 0:
            return ""
    if language == "en":
        if any(
            [
                not (
                    (ord(c) >= ord("A") and ord(c) <= ord("Z")) or c == " " or c == "'"
                )
                for c in list(sent)
            ]
        ):
            return ""
    return sent.strip()


def read_tsv(path, corpus_root, language, accent=None, hours=-1):
    with open(path, "r") as fp:
        rows = csv.reader(fp, delimiter="\t")
        data_list = []
        total_len = 0
        iterator = tqdm(enumerate(rows))
        for i, row in iterator:
            if i == 0:
                continue

            if language == "es" and row[7] != "mexicano":
                continue

            if language == "en" and row[7] != accent:
                continue

            # 0: client_id
            # 1: path
            # 2: sentence
            # 3: up_votes
            # 4: down_votes
            # 5: age
            # 6: gender
            # 7: accent
            # 8: locale
            # 9: segment

            audio = MP3(join(corpus_root, row[1]))
            secs = audio.info.length

            sent_normed = normalize(row[2], language)
            if sent_normed == "":
                continue

            data_list.append(
                {
                    "path": row[1],
                    "sentence": sent_normed,
                    "accent": row[7] if row[7] != "" else "unk",
                    "len": secs,
                }
            )
            total_len += secs

            if hours > 0 and total_len / 3600.0 > hours:
                iterator.close()
                break

        print(f"Read {len(data_list)} files")
        print("Total {:.2f} hours".format(total_len / 3600.0))

        return data_list


def write_tsv(data, out_path):
    with open(out_path, "w") as fp:
        writer = csv.writer(fp, delimiter="\t")
        writer.writerow(["path", "sentence"])
        for d in data:
            path = d["path"][:-3] + "wav"
            writer.writerow([path, d["sentence"]])


def write_txt(data, out_path):
    with open(out_path, "w") as fp:
        for d in data:
            fp.write(d["sentence"] + "\n")


def preprocess_main(pre_root, pre_lang, pre_out, pre_accent = None, pre_hours = None):
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--root", type=str, help="Root of Common Voice 7.0 directory.")
    # parser.add_argument("--lang", type=str, help="Language abbreviation.")
    # parser.add_argument("--out", type=str, help="Path to output directory.")
    # parser.add_argument("--accent", type=str, default="none", help="English accent")
    # parser.add_argument("--hours", type=float, default=-1, help="Maximum hours used.")
    # args = parser.parse_args()

    os.makedirs(pre_out, exist_ok=True)
    os.makedirs(join(pre_out, pre_lang), exist_ok=True)

    for s in ["train", "dev", "test"]:
        data_list = read_tsv(
            join(pre_root, pre_lang, s + ".tsv"),
            join(pre_root, pre_lang, "clips"),
            pre_lang,
            accent=pre_accent,
            hours=pre_hours,
        )

        if data_list[0].get("len", -1) > 0:
            data_list = sorted(data_list, reverse=True, key=lambda x: x["len"])

        write_tsv(data_list, join(pre_out, pre_lang, s + ".tsv"))

        if s == "train":
            write_txt(data_list, join(pre_out, pre_lang, s + ".txt"))


def read_processed_tsv(path):
    with open(path, "r") as fp:
        rows = csv.reader(fp, delimiter="\t")
        file_list = []
        for i, row in enumerate(rows):
            if i == 0:
                continue
            file_list.append(row[0][:-3] + "mp3")
        return file_list


def downsample_main(down_root, down_tsv):
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--root", type=str, help="Directory of the dataset.")
    # parser.add_argument("--tsv", type=str, help="Path to processed tsv file.")
    # args = parser.parse_args()

    file_list = read_processed_tsv(down_tsv)

    for file in tqdm(file_list):
        file = str(file)
        file = join(down_root, file)
        wav, sample_rate = torchaudio.load(file)
        wav = resample(
            wav.squeeze(0).numpy(), sample_rate, 16000, res_type="kaiser_best"
        )
        wav = torch.FloatTensor(wav).unsqueeze(0)
        new_file = file[:-3] + "wav"
        torchaudio.save(new_file, wav, 16000)





class CommonVoice(Dataset):
    def __init__(
        self,
        tokenizer,
        bucket_size,
        path,
        ascending=False,
        ratio=1.0,
        offset=0,
        **kwargs,
    ):
        # Preprocess
        self.cv_root=path + "/cv-corpus-7.0-2021-07-21"  # common voice 7.0 dataset location
        self.data_root=path + "common_voice" # path to save data
        # root = cv_root
        # lang = lang
        # out = data_root
        # root = cv_root + "/" + lang + "/clips"
        # tsv = data_root + "/" + lang + "/" + set + ".tsv"
        pre_lang = "zh-TW" # is temporary, in paper setting is es zh-CN ar
        preprocess_main(self.cv_root, pre_lang, self.data_root)
        for set in {"train", "dev", "test"}:
            downsample_main(self.cv_root + "/" + pre_lang + "/clips", self.data_root + "/" + pre_lang + "/" + set + ".tsv")

        split = {self.data_root + "/train", self.data_root + "/dev", self.data_root + "/test"}
        # Setup
        self.path = path
        self.bucket_size = bucket_size

        for s in split:
            with open(s, "r") as fp:
                rows = csv.reader(fp, delimiter="\t")
                file_list, text = [], []
                for i, row in enumerate(rows):
                    if i == 0:
                        continue
                    file_list.append(join(path, row[0]))
                    text.append(tokenizer.encode(row[1]))

                print(f"Found {len(file_list)} samples.")

        if ratio < 1.0:
            print(f"Ratio = {ratio}, offset = {offset}")
            skip = int(1.0 / ratio)
            file_list, text = file_list[offset::skip], text[offset::skip]
            total_len = 0.0
            for f in file_list:
                total_len += getsize(f) / 32000.0
            print(
                "Total audio len = {:.2f} mins = {:.2f} hours".format(
                    total_len / 60.0, total_len / 3600.0
                )
            )

        self.file_list, self.text = file_list, text

    def __getitem__(self, index):
        if self.bucket_size > 1:
            # Return a bucket
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
