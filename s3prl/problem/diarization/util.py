import os
import re
import torch
import logging
import subprocess
import numpy as np
from typing import List
from pathlib import Path
from scipy.signal import medfilt

logger = logging.getLogger(__name__)

RTTM_FORMAT = "SPEAKER {:s} 1 {:7.2f} {:7.2f} <NA> <NA> {:s} <NA>"


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
