#!/usr/bin/env python

# Data preparation code for the Libri2Mix dataset

import os
import argparse
import soundfile as sf
import shutil

parser = argparse.ArgumentParser(description='Prepare Libri2Mix dataset')
parser.add_argument('src_dir', type=str, help='Data directory to Libri2Mix')
parser.add_argument('tgt_dir', type=str, help='Target data directory')
parser.add_argument('--sample_rate', type=str, choices=['8k', '16k'], default='16k', help='Sample rate')
parser.add_argument('--mode', type=str, choices=['max', 'min'], default='min', help='Length mode')
parser.add_argument('--part', type=str, choices=['train-100', 'train-360', 'dev', 'test'], default='dev', help='Partition of dataset')
args = parser.parse_args()

def main():
    output_dir = "{}/wav{}/{}/{}".format(args.tgt_dir, args.sample_rate, args.mode, args.part)
    if os.path.exists(output_dir):
        raise ValueError("Warning: {} already exists, please check!".format(output_dir))
    else:
        os.makedirs(output_dir)
    
    wav_dir = "{}/wav{}/{}/{}".format(args.src_dir, args.sample_rate, args.mode, args.part)
    assert os.path.exists(wav_dir)
    for cond in ["s1", "s2", "mix_clean", "mix_both", "mix_single", "noise"]:
        if not os.path.exists("{}/{}".format(wav_dir, cond)):
            continue

        filelist = [f for f in os.listdir("{}/{}".format(wav_dir, cond)) if f.endswith(".wav")]
        filelist.sort()
        cond_dir = "{}/{}".format(output_dir, cond)
        if not os.path.exists(cond_dir):
            os.makedirs(cond_dir)
        wav_scp_file = open("{}/wav.scp".format(cond_dir), 'w')
        utt2spk_file = open("{}/utt2spk".format(cond_dir), 'w')
        for f in filelist:
            uttname = f.strip('.wav')
            wav_scp_file.write("{} {}/{}/{}\n".format(uttname, wav_dir, cond, f))
            utt2spk_file.write("{} {}\n".format(uttname, uttname))
        wav_scp_file.close()
        utt2spk_file.close()
        shutil.copyfile("{}/utt2spk".format(cond_dir), "{}/spk2utt".format(cond_dir))
    return 0

if __name__ == '__main__':
    main()
