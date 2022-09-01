#!/usr/bin/env python

# Data preparation code for the Voicebank dataset

import os
import argparse
import soundfile as sf
import shutil

parser = argparse.ArgumentParser(description='Prepare Voicebank dataset')
parser.add_argument('src_dir', type=str, help='Data directory to Voicebank')
parser.add_argument('tgt_dir', type=str, help='Target data directory')
parser.add_argument('--sample_rate', type=str, choices=['16k'], default='16k', help='Sample rate')
parser.add_argument('--part', type=str, choices=['train', 'dev', 'test'], default='dev', help='Partition of dataset')
args = parser.parse_args()

def main():
    output_dir = "{}/wav{}/{}".format(args.tgt_dir, args.sample_rate, args.part)
    if os.path.exists(output_dir):
        raise ValueError("Warning: {} already exists, please check!")
    else:
        os.makedirs(output_dir)
    
    if args.part == 'train' or args.part == 'dev':
        dset = "trainset_28spk_wav_16k"
    elif args.part == 'test':
        dset = "testset_wav_16k"
    for cond in ["clean", "noisy"]:
        wav_dir = "{}/{}_{}".format(args.src_dir, cond, dset)
        filelist = [f for f in os.listdir(wav_dir) if f.endswith(".wav")]
        filelist.sort()
        cond_dir = "{}/{}".format(output_dir, cond)
        if not os.path.exists(cond_dir):
            os.makedirs(cond_dir)
        wav_scp_file = open("{}/wav.scp".format(cond_dir), 'w')
        utt2spk_file = open("{}/utt2spk".format(cond_dir), 'w')
        for f in filelist:
            uttname = f.strip('.wav')
            if uttname.startswith('p226') or uttname.startswith('p287'):
                if args.part == 'train':
                    continue
            else:
                if args.part == 'dev':
                    continue

            wav_scp_file.write("{} {}/{}\n".format(uttname, wav_dir, f))
            utt2spk_file.write("{} {}\n".format(uttname, uttname))
        wav_scp_file.close()
        utt2spk_file.close()
        shutil.copyfile("{}/utt2spk".format(cond_dir), "{}/spk2utt".format(cond_dir))
    return 0

if __name__ == '__main__':
    main()
