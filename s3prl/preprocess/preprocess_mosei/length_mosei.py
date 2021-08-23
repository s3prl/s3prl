import os
import argparse
import pandas as pd
import numpy as np


def get_preprocess_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy_path', default='../../data/mosei/mel160', type=str, help='Path to MOSEI segmented NPY files')
    parser.add_argument('--csv_path', default='../../data/mosei/mosei_no_semi.csv', type=str, help='Path to mosei_no_semi.csv', required=False)
    args = parser.parse_args()
    return args


def add_length(args):
    csv = pd.read_csv(args.csv_path)
    lengths = []
    for index, row in csv.iterrows():
        npy = np.load(os.path.join(args.npy_path, row.key + '.npy'))
        lengths.append(npy.shape[0])
    csv['length'] = lengths
    csv.to_csv(args.csv_path, index=False)


def main():
    # get arguments
    args = get_preprocess_args()

    # Acoustic Feature Extraction
    add_length(args)


if __name__ == '__main__':
    main()

