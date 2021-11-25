# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ split_long_utter_to_short.py ]
#   Synopsis     [ preprocess long audio / speech to shorter versions ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import torch
import argparse
import torchaudio
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed
torchaudio.set_audio_backend("sox_io")


#############################
# PREPROCESS CONFIGURATIONS #
#############################
def get_preprocess_args():
    
    parser = argparse.ArgumentParser(description='preprocess arguments for any dataset.')

    parser.add_argument('-i', '--input_path', default='/livingrooms/public/LibriLight/', type=str, help='Path to your LibriSpeech directory', required=False)
    parser.add_argument('-o', '--output_path', default='/livingrooms/public/LibriLight/', type=str, help='Path to store output', required=False)
    parser.add_argument('-s', '--split_size', default=60, type=int, help='Split size in seconds', required=False)
    parser.add_argument('-a', '--audio_extension', default='.flac', type=str, help='audio file type (.wav / .flac / .mp3 / etc)', required=False)
    parser.add_argument('-n', '--name', default='-splitted', type=str, help='Name to append on the original directory', required=False)
    parser.add_argument('--n_jobs', default=-1, type=int, help='Number of jobs used for computation', required=False)

    args = parser.parse_args()
    return args


##################
# SPLIT AND SAVE #
##################
def split_and_save(input_file, current_split, args):
    wav, sr = torchaudio.load(input_file)
    
    # compute the size of each chunk
    chunk_size = args.split_size*sr
    quotient, remainder = divmod(wav.size(1), chunk_size)
    sections = [chunk_size for _ in range(quotient)]
    sections.append(remainder) # the remainder is the last chunk
    
    splitted_wav = torch.split(wav, split_size_or_sections=sections, dim=1)
    check_sum = 0
    for i, w in enumerate(splitted_wav):
        check_sum += w.size(1)
        file_name = os.path.basename(input_file).split('.')[0]
        new_file_name = file_name.replace(file_name, file_name+'-'+str(i))
        
        new_file_path = input_file.replace(current_split, current_split+args.name)
        new_file_path = new_file_path.replace(file_name, new_file_name)

        if args.input_path != args.output_path:
            new_file_path = new_file_path.replace(args.input_path, args.output_path)

        os.makedirs((os.path.dirname(new_file_path)), exist_ok=True)
        torchaudio.save(new_file_path, w, sr)
    assert check_sum == wav.size(1)


###################
# GENERATE SPLITS #
###################
def generate_splits(args, tr_set, audio_extension):
    
    for i, s in enumerate(tr_set):
        if os.path.isdir(os.path.join(args.input_path, s.lower())):
            s = s.lower()
        elif os.path.isdir(os.path.join(args.input_path, s.upper())):
            s = s.upper()
        else:
            assert NotImplementedError

        print('')
        todo = list(Path(os.path.join(args.input_path, s)).rglob('*' + audio_extension)) # '*.flac'
        print(f'Preprocessing data in: {s}, {len(todo)} audio files found.')

        print('Splitting audio to shorter length...', flush=True)
        Parallel(n_jobs=args.n_jobs)(delayed(split_and_save)(str(file), s, args) for file in tqdm(todo))

    print('All done, saved at', args.output_path, 'exit.')


########
# MAIN #
########
def main():

    # get arguments
    args = get_preprocess_args()
    
    if 'librilight' in args.input_path.lower():
        SETS = ['small', 'medium', 'large']
    elif 'librispeech' in args.input_path.lower():
        SETS = ['train-clean-100', 'train-clean-360', 'train-other-500', 'dev-clean', 'dev-other', 'test-clean', 'test-other']
    elif 'timit' in args.input_path.lower():
        SETS = ['TRAIN', 'TEST']
    else:
        raise NotImplementedError
    # change the SETS list to match your dataset, for example:
    # SETS = ['train', 'dev', 'test']
    # SETS = ['TRAIN', 'TEST']
    # SETS = ['train-clean-100', 'train-clean-360', 'train-other-500', 'dev-clean', 'dev-other', 'test-clean', 'test-other']
    
    # Select data sets
    for idx, s in enumerate(SETS):
        print('\t', idx, ':', s)
    tr_set = input('Please enter the index of splits you wish to use preprocess. (seperate with space): ')
    tr_set = [SETS[int(t)] for t in tr_set.split(' ')]

    # Run split
    generate_splits(args, tr_set, args.audio_extension)


if __name__ == '__main__':
    main()
