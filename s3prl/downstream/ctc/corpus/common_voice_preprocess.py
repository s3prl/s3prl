import argparse
import csv
import unicodedata
from string import punctuation
import re
import os
from os.path import join

from tqdm import tqdm
from mutagen.mp3 import MP3

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, help="Root of Common Voice 7.0 directory.")
    parser.add_argument("--lang", type=str, help="Language abbreviation.")
    parser.add_argument("--out", type=str, help="Path to output directory.")
    parser.add_argument("--accent", type=str, default="none", help="English accent")
    parser.add_argument("--hours", type=float, default=-1, help="Maximum hours used.")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(join(args.out, args.lang), exist_ok=True)

    for s in ["train", "dev", "test"]:
        data_list = read_tsv(
            join(args.root, args.lang, s + ".tsv"),
            join(args.root, args.lang, "clips"),
            args.lang,
            accent=args.accent,
            hours=args.hours,
        )

        if data_list[0].get("len", -1) > 0:
            data_list = sorted(data_list, reverse=True, key=lambda x: x["len"])

        write_tsv(data_list, join(args.out, args.lang, s + ".tsv"))

        if s == "train":
            write_txt(data_list, join(args.out, args.lang, s + ".txt"))


if __name__ == "__main__":
    main()
