import os
import random
import argparse

parser = argparse.ArgumentParser(description='Subsample data for Libri2Mix')
parser.add_argument('src_dir', type=str, help='Source directory')
parser.add_argument('tgt_dir', type=str, help='Target directory')
parser.add_argument('--sample', type=int, default=1000, help='Number of samples')
parser.add_argument('--seed', type=int, default=7, help='Random seed')
args = parser.parse_args()

def get_utt2path(wav_scp_file):
    utt2path = {}
    with open(wav_scp_file, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        utt, path = line.split()
        utt2path[utt] = path
    return utt2path

def main():
    # sampling the audio files
    random.seed(args.seed)
    with open("{}/s1/utt2spk".format(args.src_dir), 'r') as fh:
        content = fh.readlines()
    uttlist = []
    for line in content:
        line = line.strip('\n')
        utt = line.split()[0]
        uttlist.append(utt)
    uttlist.sort()
    num_utt_ori = len(uttlist)
    random.shuffle(uttlist)
    uttlist = uttlist[:args.sample]
    uttlist.sort()
    print("Selecting {} utts from {} utts".format(len(uttlist), num_utt_ori))

    # mix_both, mix_clean, mix_single, noise, s1, s2
    for dset in ["mix_both", "mix_clean", "mix_single", "noise", "s1", "s2"]:
        src_dset, tgt_dset = "{}/{}".format(args.src_dir, dset), "{}/{}".format(args.tgt_dir, dset)
        os.makedirs(tgt_dset)
        with open("{}/utt2spk".format(tgt_dset), 'w') as fh:
            for utt in uttlist:
                fh.write("{} {}\n".format(utt, utt))
        with open("{}/spk2utt".format(tgt_dset), 'w') as fh:
            for utt in uttlist:
                fh.write("{} {}\n".format(utt, utt))
        utt2path = get_utt2path("{}/wav.scp".format(src_dset))
        with open("{}/wav.scp".format(tgt_dset), 'w') as fh:
            for utt in uttlist:
                fh.write("{} {}\n".format(utt, utt2path[utt]))
    return 0

if __name__ == '__main__':
    main()
