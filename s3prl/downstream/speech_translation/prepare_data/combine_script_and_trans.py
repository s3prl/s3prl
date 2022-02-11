import pandas as pd
import argparse
import csv

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--transcript-file', required=True)
    parser.add_argument('-t', '--translation-tsv', required=True)
    parser.add_argument('-o', '--output-tsv', required=True)
    parser.add_argument('--translation-id-key', default='id')
    parser.add_argument('--translation-trans-key', default='tgt_text')
    parser.add_argument('--id-key', default='id')
    parser.add_argument('--src-key', default='src_text')
    parser.add_argument('--tgt-key', default='tgt_text')
    parser.add_argument('--path-key', default='path')

    args = parser.parse_args()

    trans_data = pd.read_csv(
        args.translation_tsv,
        sep='\t',
        index_col=args.translation_id_key,
        quoting=csv.QUOTE_NONE
    )

    output = {
        args.id_key: [],
        args.path_key: [],
        args.src_key: [],
        args.tgt_key: [],
    }

    for line in open(args.transcript_file):
        
        index, transcript = line.strip().split(maxsplit=1)
        
        translation = trans_data.at[index, args.translation_trans_key]
        
        output[args.id_key].append(index)
        output[args.path_key].append(f'{index}.wav')
        output[args.src_key].append(transcript)
        output[args.tgt_key].append(translation)
    
    output_df = pd.DataFrame(
        data=output,
    )

    output_df.to_csv(
        args.output_tsv,
        sep='\t',
        doublequote=False,
        quoting=csv.QUOTE_NONE,
        index=False,
    )