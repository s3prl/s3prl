import csv
import argparse
import string
from tqdm.auto import tqdm
import os
import torchaudio

def verbose(args, text):
    if args.verbose:
        print(text)

def length(s):
    return len(s.split())

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input_tsv')
    parser.add_argument('output_tsv')
    parser.add_argument('-s', '--src-key', default='src_text')
    parser.add_argument('-t', '--tgt-key', default='tgt_text')
    parser.add_argument('-o', '--overwrite', action='store_true')
    parser.add_argument('-u', '--max', type=int, default=-1)
    parser.add_argument('-l', '--min', type=int, default=-1)
    parser.add_argument('-r', '--ratio', type=float, default=-1)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    if os.path.isfile(args.output_tsv) and not args.overwrite:
        print(f'output file: {args.output_tsv} exists, use -o/--overwrite to force overwrite')
        exit(1)
    verbose(args, args)
    
    if args.ratio >= 0:
        assert args.min >= 1, "minimum length should >= 1 with ratio test"
    
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

    data = []
    for line in tqdm(lines):

        src_len = length(line[args.src_key])
        tgt_len = length(line[args.tgt_key])

        if 'REMOVE' in line[args.src_key] or 'REMOVE' in line[args.tgt_key]:
            verbose(args, f"{line} contains \"REMOVE\", skip")
            continue

        if args.max >= 0:
            if src_len > args.max or tgt_len > args.max:
                verbose(args, f"{line} text part too long, skip")
                continue
        
        if args.min >= 0:
            if src_len < args.min or tgt_len < args.min:
                verbose(args, f"{line} text part too short, skip")
                continue

        if args.ratio > 0:
            if src_len / tgt_len > args.ratio or tgt_len / src_len > args.ratio:
                verbose(args, f"{line} text part invalid ratio, skip")
                continue

        data.append(line)

    print(f'remove {len(lines)-len(data)}/{len(lines)} samples')

    with open(args.output_tsv, 'w') as f:
        writer = csv.DictWriter(
            f,
            delimiter='\t',
            quotechar=None,
            doublequote=False,
            lineterminator='\n',
            quoting=csv.QUOTE_NONE,
            fieldnames=data[0].keys(),
        )
        writer.writeheader()
        writer.writerows(data)
