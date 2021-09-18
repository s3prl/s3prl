import csv
import argparse
import string
from tqdm.auto import tqdm
import os
import sacremoses

def verbose(args, text):
    if args.verbose:
        print(text)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input_tsv')
    parser.add_argument('output_tsv')
    parser.add_argument('key')
    parser.add_argument('-n', '--normalize', action='store_true')
    parser.add_argument('-l', '--lowercase', action='store_true')
    parser.add_argument('-r', '--remove-punctuation', action='store_true')
    parser.add_argument('-L', '--lang', default='en')
    parser.add_argument('-o', '--overwrite', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    
    if os.path.isfile(args.output_tsv) and not args.overwrite:
        print(f'output file: {args.output_tsv} exists, use -o/--overwrite to force overwrite')
        exit(1)

    verbose(args, args)

    normalizer = sacremoses.MosesPunctNormalizer(
        lang=args.lang,
        pre_replace_unicode_punct=True,
        post_remove_control_chars=True,
    )
    p_list = set(string.punctuation) - set("'-")

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

        text = line[args.key]

        if args.normalize:
            text = normalizer.normalize(text)
            for c in ".?,'":
                text = text.replace(f' {c}', c)
            text = text.replace('do n\'t', 'don\'t')
        if args.remove_punctuation:
            text = ''.join([ c if c not in p_list else '' for c in text])
            text = ' '.join(text.split())
        if args.lowercase:
            text = text.lower()
        line[args.key] = text
        data.append(line)

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