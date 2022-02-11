import pandas as pd
import argparse
import csv

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-tsv', required=True)
    parser.add_argument('--id-key', required=True)
    parser.add_argument('--id-file', required=True)
    parser.add_argument('--output-tsv', required=True)
    args = parser.parse_args()

    raw_data = pd.read_csv(args.input_tsv,
        sep='\t',
        index_col=args.id_key,
        quoting=csv.QUOTE_NONE
    )

    indexs = []
    for line in open(args.id_file):
        index = line.strip()
        assert index in raw_data.index
        indexs.append(index)

    selected_data = raw_data.loc[indexs]

    selected_data.to_csv(args.output_tsv,
        sep='\t',
        doublequote=False,
        quoting=csv.QUOTE_NONE,
    )