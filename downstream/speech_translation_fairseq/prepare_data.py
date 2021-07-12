import csv
import torchaudio
from tqdm.auto import tqdm
import os
import sys

if __name__ == '__main__':

    src_lang = 'en'
    tgt_lang = 'de'
    # covost_root = '/hdd/covost/cv-corpus-6.1-2020-12-11'
    covost_root = '/home/sean/battleship/A'
    tsv_dir = '/hdd/covost/tsv'
    output_dir = '/home/sean/battleship/s3prl/data/test/'

    split = sys.argv[1]

    data = []
    with open(f'{tsv_dir}/covost_v2.{src_lang}_{tgt_lang}.{split}.tsv') as file:

        lines = []
        reader = csv.DictReader(
            file,
            delimiter="\t",
            quotechar=None,
            doublequote=False,
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
        )
        lines = []
        for line in reader:
            lines.append(line)

        for line in tqdm(lines):
            # file_path = f"{covost_root}/{src_lang}/clips/{line['path']}"
            file_path = f"{covost_root}/{line['path']}"
            wav, sr = torchaudio.load(file_path)
            item = {
                'id': line['path'].split('.')[0],
                'audio': line['path'],
                'n_frames': wav.size(1),
                'sr': sr,
                'src_text': line['sentence'],
                'tgt_text': line['translation']
            }
            data.append(item)
        
        data.sort(key=lambda x: x['n_frames'])

    os.makedirs(output_dir, exist_ok=True)


    with open(f'{output_dir}/{split}_st_{src_lang}_{tgt_lang}.tsv', 'w') as file:
        writer = csv.DictWriter(file,
                delimiter='\t',
                quotechar=None,
                doublequote=False,
                lineterminator="\n",
                quoting=csv.QUOTE_NONE,
                fieldnames=['id', 'audio', 'n_frames', 'sr', 'src_text', 'tgt_text'])
        writer.writeheader()
        writer.writerows(data)