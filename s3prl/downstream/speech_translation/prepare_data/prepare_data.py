import csv
import argparse
import string
from tqdm.auto import tqdm
import os
import torchaudio
from multiprocessing import Pool

def verbose(args, text):
    if args.verbose:
        print(text)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input_tsv')
    parser.add_argument('output_tsv')
    parser.add_argument('-p', '--path-key', default='path')
    parser.add_argument('-s', '--src-key', default='src_text')
    parser.add_argument('-t', '--tgt-key', default='tgt_text')
    parser.add_argument('-d', '--audio-dir', default='.')
    parser.add_argument('-o', '--overwrite', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    if os.path.isfile(args.output_tsv) and not args.overwrite:
        print(f'output file: {args.output_tsv} exists, use -o/--overwrite to force overwrite')
        exit(1)
    verbose(args, args)
    data = []
    lines = []
    with open(args.input_tsv, 'r') as f:
        reader = csv.DictReader(
            f,
            delimiter='\t',
            quotechar=None,
            doublequote=False,
            lineterminator='\n',
            quoting=csv.QUOTE_NONE,
        )
        for line in reader:
            lines.append(line)

    for line in tqdm(lines):

        if line[args.path_key] == None:
            verbose(args, f"{line} path not provide, skip")
        file_path = os.path.join(args.audio_dir, line[args.path_key])
        if not os.path.isfile(file_path):
            verbose(args, f"{file_path} not exists, skip")
            continue
        wav, sr = torchaudio.load(file_path)
        item = {
            'id': line[args.path_key].split('.')[0],
            'audio': line[args.path_key],
            'n_frames': wav.size(1),
            'sr': sr,
            'src_text': line[args.src_key],
            'tgt_text': line[args.tgt_key],
        }
        data.append(item)
    
    data.sort(key=lambda x: x['n_frames'])

    with open(args.output_tsv, 'w') as f:
        writer = csv.DictWriter(
            f,
            delimiter='\t',
            quotechar=None,
            doublequote=False,
            lineterminator='\n',
            quoting=csv.QUOTE_NONE,
            fieldnames=['id', 'audio', 'n_frames', 'sr', 'src_text', 'tgt_text']
        )
        writer.writeheader()
        writer.writerows(data)