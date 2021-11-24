# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ evaluate.py ]
#   Synopsis     [ main objective evaluation script for any-to-one voice conversion ]
#   Author       [ Wen-Chin Huang (https://github.com/unilight) ]
#   Copyright    [ Copyright(c), Toda Lab, Nagoya University, Japan ]
"""*********************************************************************************************"""


import argparse
import multiprocessing as mp
import os

import numpy as np
import librosa

import torch
from tqdm import tqdm
import yaml

from utils import find_files
from vc_evaluate import calculate_mcd_f0
from vc_evaluate import load_asr_model, transcribe, calculate_measures
from vc_evaluate import load_asv_model, calculate_threshold, calculate_accept

def get_basename(path):
    return os.path.splitext(os.path.split(path)[-1])[0]

def get_number(basename):
    # converted: <srcspk>_E<number>_gen
    if "_" in basename:
        return basename.split("_")[1]
    # ground truth: E<number>
    else:
        return basename

def _calculate_asv_score(model, file_list, gt_root, trgspk, threshold):
    results = {}
    for i, cvt_wav_path in enumerate(tqdm(file_list)):
        basename = get_basename(cvt_wav_path)
        number = get_number(basename)
        
        # get ground truth target wav path
        gt_wav_path = os.path.join(gt_root, trgspk, number + ".wav")

        # calculate acept
        results[basename] = calculate_accept(cvt_wav_path, gt_wav_path, model, threshold)

    return results, 100.0 * float(np.mean(np.array(list(results.values()))))

def _calculate_asr_score(model, device, file_list, groundtruths):
    keys = ["hits", "substitutions",  "deletions", "insertions"]
    ers = {}
    c_results = {k: 0 for k in keys}
    w_results = {k: 0 for k in keys}

    for i, cvt_wav_path in enumerate(tqdm(file_list)):
        basename = get_basename(cvt_wav_path)
        number = get_number(basename)
        groundtruth = groundtruths[number[1:]] # get rid of the first character "E"
        
        # load waveform
        wav, _ = librosa.load(cvt_wav_path, sr=16000)

        # trascribe
        transcription = transcribe(model, device, wav)

        # error calculation
        c_result, w_result, norm_groundtruth, norm_transcription = calculate_measures(groundtruth, transcription)

        ers[basename] = [c_result["cer"] * 100.0, w_result["wer"] * 100.0, norm_transcription, norm_groundtruth]

        for k in keys:
            c_results[k] += c_result[k]
            w_results[k] += w_result[k]
  
    # calculate over whole set
    def er(r):
        return float(r["substitutions"] + r["deletions"] + r["insertions"]) \
            / float(r["substitutions"] + r["deletions"] + r["hits"]) * 100.0

    cer = er(c_results)
    wer = er(w_results)

    return ers, cer, wer

def _calculate_mcd_f0(file_list, gt_root, trgspk, f0min, f0max, results):
    for i, cvt_wav_path in enumerate(file_list):
        basename = get_basename(cvt_wav_path)
        number = get_number(basename)
        
        # get ground truth target wav path
        gt_wav_path = os.path.join(gt_root, trgspk, number + ".wav")

        # read both converted and ground truth wav
        cvt_wav, cvt_fs = librosa.load(cvt_wav_path, sr=None)
        gt_wav, gt_fs = librosa.load(gt_wav_path, sr=None)
        assert cvt_fs == gt_fs

        # calculate MCD, F0RMSE, F0CORR and DDUR
        mcd, f0rmse, f0corr, ddur = calculate_mcd_f0(cvt_wav, gt_wav, gt_fs, f0min, f0max)

        results.append([basename, mcd, f0rmse, f0corr, ddur])

def get_parser():
    parser = argparse.ArgumentParser(description="objective evaluation script.")
    parser.add_argument("--wavdir", required=True, type=str, help="directory for converted waveforms")
    parser.add_argument("--trgspk", required=True, type=str, help="target speaker")
    parser.add_argument("--data_root", type=str, default="./data", help="directory of data")
    parser.add_argument("--log_path", type=str, default=None,
                         help="path of output log. If not specified, output to <wavdir>/obj.log")
    parser.add_argument("--n_jobs", default=10, type=int, help="number of parallel jobs")
    return parser


def main():
    args = get_parser().parse_args()

    trgspk = args.trgspk
    task = "task1" if trgspk[1] == "E" else "task2"
    gt_root = os.path.join(args.data_root, "vcc2020")
    f0_path = os.path.join(args.data_root, "f0.yaml")
    threshold_path = os.path.join(args.data_root, "thresholds.yaml")
    transcription_path = os.path.join(args.data_root, "vcc2020", "prompts", "Eng_transcriptions.txt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # load f0min and f0 max
    with open(f0_path, 'r') as f:
        f0_all = yaml.load(f, Loader=yaml.FullLoader)
    f0min = f0_all[trgspk]["f0min"]
    f0max = f0_all[trgspk]["f0max"]

    # load ground truth transcriptions
    with open(transcription_path, "r") as f:
        lines = f.read().splitlines()
    groundtruths = {line.split(" ")[0]: " ".join(line.split(" ")[1:]) for line in lines}

    # find converted files
    converted_files = sorted(find_files(args.wavdir, query="*300??*.wav"))
    print("number of utterances = {}".format(len(converted_files)))

    # load threshold; if does not exist, calculate
    threshold = None
    threshold_all = {}
    if os.path.exists(threshold_path):
        with open(threshold_path, 'r') as f:
            threshold_all = yaml.load(f, Loader=yaml.FullLoader)
            if threshold_all and task in threshold_all:
                equal_error_rate, threshold = threshold_all[task]
    if not threshold:
        equal_error_rate, threshold = calculate_threshold(gt_root, task, device)
        if threshold_all:
            threshold_all[task] = [equal_error_rate, threshold]
        else:
            threshold_all = {task: [equal_error_rate, threshold]}
        with open(threshold_path, 'w') as f:
            yaml.safe_dump(threshold_all, f)
    print(f"[INFO]: Equal error rate: {equal_error_rate}")
    print(f"[INFO]: Threshold: {threshold}")

    ##############################

    print("Calculating ASV-based score...")
    # load ASV model
    asv_model = load_asv_model(device)

    # calculate accept rate
    accept_results, accept_rate = _calculate_asv_score(asv_model, converted_files, gt_root, trgspk, threshold)

    ##############################

    print("Calculating ASR-based score...")
    # load ASR model
    asr_model = load_asr_model(device)

    # calculate error rates
    ers, cer, wer = _calculate_asr_score(asr_model, device, converted_files, groundtruths)
    
    ##############################

    if task == "task1":
        print("Calculating MCD and f0-related scores...")
        # Get and divide list
        file_lists = np.array_split(converted_files, args.n_jobs)
        file_lists = [f_list.tolist() for f_list in file_lists]

        # multi processing
        with mp.Manager() as manager:
            results = manager.list()
            processes = []
            for f in file_lists:
                p = mp.Process(
                    target=_calculate_mcd_f0,
                    args=(f, gt_root, trgspk, f0min, f0max, results),
                )
                p.start()
                processes.append(p)

            # wait for all process
            for p in processes:
                p.join()

            results = sorted(results, key=lambda x:x[0])
            results = [result + ers[result[0]] + [accept_results[result[0]]] for result in results] 
    else:
        results = []
        for f in converted_files:
            basename = get_basename(f)
            results.append([basename] + ers[basename] + [accept_results[basename]])
        
    # write to log
    log_path = args.log_path if args.log_path else os.path.join(args.wavdir, "obj.log")
    with open(log_path, "w") as f:
        
        # average result
        if task == "task1":
            mMCD = np.mean(np.array([result[1] for result in results]))
            mf0RMSE = np.mean(np.array([result[2] for result in results]))
            mf0CORR = np.mean(np.array([result[3] for result in results]))
            mDDUR = np.mean(np.array([result[4] for result in results]))
        mCER = cer 
        mWER = wer
        mACCEPT = accept_rate

        # utterance wise result
        for result in results:
            if task == "task1":
                f.write(
                    "{} {:.2f} {:.2f} {:.2f} {:.2f} {:.1f} {:.1f} {} \t{} | {}\n".format(
                        *result
                    )
                )
            elif task == "task2":
                f.write(
                    "{} {:.1f} {:.1f} {} \t{} | {}\n".format(
                        *result
                    )
                )

        if task == "task1":
            print(
                "Mean MCD, f0RMSE, f0CORR, DDUR, CER, WER, accept rate: {:.2f} {:.2f} {:.3f} {:.3f} {:.1f} {:.1f} {:.2f}".format(
                    mMCD, mf0RMSE, mf0CORR, mDDUR, mCER, mWER, mACCEPT
                )
            )
            f.write(
                "Mean MCD, f0RMSE, f0CORR, DDUR, CER, WER, accept rate: {:.2f} {:.2f} {:.3f} {:.3f} {:.1f} {:.1f} {:.2f}".format(
                    mMCD, mf0RMSE, mf0CORR, mDDUR, mCER, mWER, mACCEPT
                )
            )
        elif task == "task2":
            print(
                "Mean CER, WER, accept rate: {:.1f} {:.1f} {:.2f}".format(
                    mCER, mWER, mACCEPT
                )
            )
            f.write(
                "Mean CER, WER, accept rate: {:.1f} {:.1f} {:.2f}".format(
                     mCER, mWER, mACCEPT
                )
            )


if __name__ == "__main__":
    main()
