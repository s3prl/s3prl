"""
Utilities for file format conversion for Speaker Diarization

Authors:
  * Jiatong Shi 2021
  * Leo 2022
"""

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from scipy.signal import medfilt
from tqdm import tqdm

logger = logging.getLogger(__name__)

RTTM_FORMAT = "SPEAKER {:s} 1 {:7.2f} {:7.2f} <NA> <NA> {:s} <NA>"

__all__ = [
    "kaldi_dir_to_rttm",
    "csv_to_kaldi_dir",
    "kaldi_dir_to_csv",
]


def kaldi_dir_to_rttm(data_dir: str, rttm_path: str):
    data_dir: Path = Path(data_dir)
    segments_file = data_dir / "segments"
    utt2spk_file = data_dir / "utt2spk"

    assert segments_file.is_file()
    assert utt2spk_file.is_file()

    utt2spk = {}
    with utt2spk_file.open() as f:
        for utt2spk_line in f.readlines():
            fields = utt2spk_line.strip().replace("\n", " ").split()
            assert len(fields) == 2
            utt, spk = fields
            utt2spk[utt] = spk

    with Path(rttm_path).open("w") as rttm_f:
        with segments_file.open() as f:
            for segment_line in f.readlines():
                fields = segment_line.strip().replace("\t", " ").split()
                assert len(fields) == 4
                utt, reco, start, end = fields
                spk = utt2spk[utt]
                print(
                    RTTM_FORMAT.format(
                        reco,
                        float(start),
                        float(end) - float(start),
                        spk,
                    ),
                    file=rttm_f,
                )


def make_rttm_and_score(
    prediction_dir: str,
    score_dir: str,
    gt_rttm: str,
    frame_shift: int,
    thresholds: List[int],
    medians: List[int],
    subsampling: int = 1,
    sampling_rate: int = 16000,
):
    Path(score_dir).mkdir(exist_ok=True, parents=True)
    dscore_dir = Path(score_dir) / "dscore"
    rttm_dir = Path(score_dir) / "rttm"
    result_dir = Path(score_dir) / "result"

    setting2dscore = []
    for th in thresholds:
        for med in medians:
            logger.info(f"Make RTTM with threshold {th}, median filter {med}")
            rttm_file = rttm_dir / f"threshold-{th}_median-{med}.rttm"
            make_rttm(
                prediction_dir,
                rttm_file,
                th,
                med,
                frame_shift,
                subsampling,
                sampling_rate,
            )

            logger.info(f"Scoring...")
            result_file = result_dir / f"threshold-{th}_median-{med}.result"
            overall_der = score_with_dscore(dscore_dir, rttm_file, gt_rttm, result_file)
            logger.info(f"DER: {overall_der}")

            setting2dscore.append(((th, med), overall_der))

    setting2dscore.sort(key=lambda x: x[1])
    (best_th, best_med), best_der = setting2dscore[0]
    return best_der, (best_th, best_med)


def make_rttm(
    prediction_dir: str,
    out_rttm_path: str,
    threshold: int,
    median: int,
    frame_shift: int,
    subsampling: int,
    sampling_rate: int,
):
    names = sorted([name for name in os.listdir(prediction_dir)])
    filepaths = [Path(prediction_dir) / name for name in names]

    Path(out_rttm_path).parent.mkdir(exist_ok=True, parents=True)
    with open(out_rttm_path, "w") as wf:
        for filepath in filepaths:
            session, _ = os.path.splitext(os.path.basename(filepath))
            data = torch.load(filepath).numpy()
            a = np.where(data > threshold, 1, 0)
            if median > 1:
                a = medfilt(a, (median, 1))
            factor = frame_shift * subsampling / sampling_rate
            for spkid, frames in enumerate(a.T):
                frames = np.pad(frames, (1, 1), "constant")
                (changes,) = np.where(np.diff(frames, axis=0) != 0)
                for s, e in zip(changes[::2], changes[1::2]):
                    print(
                        RTTM_FORMAT.format(
                            session,
                            s * factor,
                            (e - s) * factor,
                            session + "_" + str(spkid),
                        ),
                        file=wf,
                    )


def score_with_dscore(
    dscore_dir: str, hyp_rttm: str, gt_rttm: str, score_result: str
) -> float:
    """
    This function returns the overall DER score, and will also write the detailed scoring results
    to 'score_result'
    """
    dscore_dir: Path = Path(dscore_dir)
    Path(score_result).parent.mkdir(exist_ok=True, parents=True)

    if not dscore_dir.is_dir():
        logger.info(f"Cloning dscore into {dscore_dir}")
        subprocess.check_output(
            f"git clone https://github.com/nryant/dscore.git {dscore_dir}",
            shell=True,
        ).decode("utf-8")

    subprocess.check_call(
        f"python3 {dscore_dir}/score.py -r {gt_rttm} -s {hyp_rttm} > {score_result}",
        shell=True,
    )

    return get_overall_der_from_dscore_file(score_result)


def get_overall_der_from_dscore_file(score_result: str):
    with open(score_result) as file:
        lines = file.readlines()
        overall_lines = [line for line in lines if "OVERALL" in line]
        assert len(overall_lines) == 1
        overall_line = overall_lines[0]
        overall_line = re.sub("\t+", " ", overall_line)
        overall_line = re.sub(" +", " ", overall_line)
        overall_der = float(overall_line.split(" ")[3])
        # The overall der line should look like:
        # *** OVERALL *** DER JER ...
    return overall_der


def csv_to_kaldi_dir(csv: str, data_dir: str):
    logger.info(f"Convert csv {csv} into kaldi data directory {data_dir}")

    data_dir: Path = Path(data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(csv)
    required = ["record_id", "wav_path", "utt_id", "speaker", "start_sec", "end_sec"]
    for r in required:
        assert r in df.columns

    reco2path = {}
    reco2dur = {}
    utt2spk = {}
    spk2utt = {}
    segments = []
    for rowid, row in tqdm(df.iterrows(), total=len(df)):
        record_id, wav_path, duration, utt_id, speaker, start_sec, end_sec = (
            row["record_id"],
            row["wav_path"],
            row["duration"],
            row["utt_id"],
            row["speaker"],
            row["start_sec"],
            row["end_sec"],
        )
        if record_id in reco2path:
            assert wav_path == reco2path[record_id]
        else:
            reco2path[record_id] = wav_path

        if record_id not in reco2dur:
            reco2dur[record_id] = duration
        else:
            assert reco2dur[record_id] == duration

        if utt_id not in utt2spk:
            utt2spk[utt_id] = str(speaker)
        else:
            assert utt2spk[utt_id] == str(speaker)

        if speaker not in spk2utt:
            spk2utt[speaker] = []
        spk2utt[speaker].append(utt_id)

        segments.append((utt_id, record_id, str(start_sec), str(end_sec)))

    with (data_dir / "wav.scp").open("w") as f:
        f.writelines([f"{reco} {path}\n" for reco, path in reco2path.items()])

    with (data_dir / "reco2dur").open("w") as f:
        f.writelines([f"{reco} {dur}\n" for reco, dur in reco2dur.items()])

    with (data_dir / "utt2spk").open("w") as f:
        f.writelines([f"{utt} {spk}\n" for utt, spk in utt2spk.items()])

    with (data_dir / "spk2utt").open("w") as f:
        f.writelines([f"{spk} {' '.join(utts)}\n" for spk, utts in spk2utt.items()])

    with (data_dir / "segments").open("w") as f:
        f.writelines(
            [f"{utt} {record} {start} {end}\n" for utt, record, start, end in segments]
        )


def kaldi_dir_to_csv(data_dir: str, csv: str):
    logger.info(f"Convert kaldi data directory {data_dir} into csv {csv}")

    data_dir: Path = Path(data_dir)

    assert (data_dir / "wav.scp").is_file()
    assert (data_dir / "segments").is_file()
    assert (data_dir / "utt2spk").is_file()
    assert (data_dir / "reco2dur").is_file()

    reco2path = {}
    with (data_dir / "wav.scp").open() as f:
        for line in f.readlines():
            line = line.strip()
            reco, path = line.split(" ")
            reco2path[reco] = path

    reco2dur = {}
    with (data_dir / "reco2dur").open() as f:
        for line in f.readlines():
            line = line.strip()
            reco, duration = line.split(" ")
            reco2dur[reco] = float(duration)

    utt2spk = {}
    with (data_dir / "utt2spk").open() as f:
        for line in f.readlines():
            line = line.strip()
            utt, spk = line.split(" ")
            utt2spk[utt] = spk

    row = []
    with (data_dir / "segments").open("r") as f:
        for line in f.readlines():
            line = line.strip()
            utt, reco, start, end = line.split(" ")
            row.append(
                (
                    reco,
                    reco2path[reco],
                    reco2dur[reco],
                    utt,
                    utt2spk[utt],
                    float(start),
                    float(end),
                )
            )

    recos, wav_paths, durations, utts, spks, starts, ends = zip(*row)
    pd.DataFrame(
        data=dict(
            record_id=recos,
            wav_path=wav_paths,
            utt_id=utts,
            speaker=spks,
            start_sec=starts,
            end_sec=ends,
            duration=durations,
        )
    ).to_csv(csv, index=False)
