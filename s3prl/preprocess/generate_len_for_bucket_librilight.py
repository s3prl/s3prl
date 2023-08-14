###############
# IMPORTATION #
###############
# wget --show-progress https://dl.fbaipublicfiles.com/librilight/data/librispeech_finetuning.tgz
import os
import argparse
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed


#############################
# PREPROCESS CONFIGURATIONS #
#############################
def get_preprocess_args():

    parser = argparse.ArgumentParser(description='preprocess arguments for any dataset.')

    parser.add_argument('-i', '--input_data', default='../LibriLight/', type=str, help='Path to your LibriLight directory', required=False)
    parser.add_argument('-o', '--output_path', default='./data/LibriLight/', type=str, help='Path to store output', required=False)
    parser.add_argument('-a', '--audio_extension', default='.flac', type=str, help='audio file type (.wav / .flac / .mp3 / etc)', required=False)
    parser.add_argument('-n', '--name', default='len_for_bucket', type=str, help='Name of the output directory', required=False)
    parser.add_argument('--n_jobs', default=-1, type=int, help='Number of jobs used for feature extraction', required=False)

    args = parser.parse_args()
    return args


##################
# EXTRACT LENGTH #
##################
def extract_length(input_file):
    torchaudio.set_audio_backend("sox_io")
    return torchaudio.info(input_file).num_frames


###################
# GENERATE LENGTH #
###################
def generate_length(args, tr_set, audio_extension):

    n1h_dir = os.path.join(args.input_data, '1h')
    n9h_dir = os.path.join(args.input_data, '9h')
    n10min_dir = os.path.join(n1h_dir, '0') # 0 is the first sub_dir

    assert os.path.exists(n1h_dir) \
           and os.path.exists(n9h_dir) \
           and os.path.exists(n10min_dir) \
           , f'Please download LibriLight dataset and put it in {args.input_data}'

    def get_audio_list(path):
        return list(Path(path).rglob('*' + audio_extension)) # '*.flac'

    for i, s in enumerate(tr_set):
        todo = []
        match s:
            case '10min':
                todo += get_audio_list(n10min_dir)
            case '1h':
                todo += get_audio_list(n1h_dir)
            case '10h':
                todo += get_audio_list(n1h_dir)
                todo += get_audio_list(n9h_dir)
            case _:
                raise NotImplementedError

        print('')
        print(f'Preprocessing data in: {s}, {len(todo)} audio files found.')

        output_dir = os.path.join(args.output_path, args.name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print('Extracting audio length...', flush=True)
        tr_x = Parallel(n_jobs=args.n_jobs)(delayed(extract_length)(str(file)) for file in tqdm(todo))
        sorted_idx = list(reversed(np.argsort(tr_x)))

        # sort by len
        sorted_todo = [os.path.relpath(str(todo[idx]), args.input_data) for idx in sorted_idx]
        sorted_tr_x = [tr_x[idx] for idx in sorted_idx]

        # Dump data
        lengths = sorted_tr_x
        print('Total:{}mins, Min:{}secs, Max:{}secs'.format(sum(lengths)//960000, min(lengths or [0])//16000, max(lengths or [0])//16000))
        df = pd.DataFrame(data={'file_path':sorted_todo, 'length':sorted_tr_x, 'label':None})
        df.to_csv(os.path.join(output_dir, tr_set[i] + '.csv'))

    print('All done, saved at', output_dir, 'exit.')


########
# MAIN #
########
def main():

    # get arguments
    args = get_preprocess_args()

    SETS = ['10min', '1h', '10h']

    # Select data sets
    for idx, s in enumerate(SETS):
        print('\t', idx, ':', s)
    tr_set = input('Please enter the index of splits you wish to use preprocess. (seperate with space): ')
    tr_set = [SETS[int(t)] for t in tr_set.split(' ')]

    # Acoustic Feature Extraction & Make Data Table
    generate_length(args, tr_set, args.audio_extension)


if __name__ == '__main__':
    main()
