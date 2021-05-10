import os
import glob
import argparse
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from joblib import Parallel, delayed

parser = argparse.ArgumentParser()
parser.add_argument('--libri', help='The root directory of Librispeech')
parser.add_argument('--lexicon', help='The lexicon given by LibriSpeech official website')
parser.add_argument('--output_words', help='The words not in official lexicon')
parser.add_argument('--njobs', type=int, default=12)
args = parser.parse_args()

words_in_lexicon = defaultdict(lambda: False)
with open(args.lexicon, 'r') as file:
    lines = [line[:-1].replace('\t', ' ') for line in file.readlines()]
    for line in lines:
        words_in_lexicon[line.split()[0]] = True

assert os.path.isdir(args.libri)
all_flac = Path(args.libri).rglob('*.flac')

def locate_txt(flac):
    filename = os.path.basename(flac)
    tags = filename.split('.')[0].split('-')
    txt_path = os.path.join(os.path.dirname(flac), f'{tags[0]}-{tags[1]}.trans.txt')
    return txt_path    

all_txt = Parallel(n_jobs=args.njobs)(delayed(locate_txt)(flac) for flac in tqdm(list(all_flac), desc='locate txt'))
all_txt = set(all_txt)

words_not_in_lexicon = []
for txt_path in tqdm(all_txt, desc='find words'):
    with open(txt_path, 'r') as file:
        lines = [line[:-1].replace('\t', ' ') for line in file.readlines()]
        for line in lines:
            idx, transcription = line.split(' ', 1)
            for word in transcription.split():
                if not words_in_lexicon[word]:
                    words_not_in_lexicon.append(f'{word}\n')

words_not_in_lexicon = set(words_not_in_lexicon)
print(f'{len(words_not_in_lexicon)} words not found in LibriSpeech lexicon.')

with open(args.output_words, 'w') as file:
    file.writelines(words_not_in_lexicon)
