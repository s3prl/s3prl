import argparse
import csv
from pathlib import Path
from sacrebleu.metrics import BLEU

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-dir', required=True)
    parser.add_argument('--tsv-file', default='output-st-test.tsv')
    parser.add_argument('--hyp-key', default='hyp')
    parser.add_argument('--ref-key', default='ref')
    args = parser.parse_args()

    args.exp_dir = Path(args.exp_dir)
    hyps, refs = [], []

    with open(args.exp_dir/args.tsv_file, 'r') as f:
        reader = csv.DictReader(
            f,
            delimiter='\t',
            quotechar=None,
            doublequote=False,
            lineterminator='\n',
            quoting=csv.QUOTE_NONE,
        )
        for line in reader:
            hyps.append(line[args.hyp_key])
            refs.append(line[args.ref_key])

    bleu = BLEU()
    score = bleu.corpus_score(hyps, [refs])
    print(score.score)
            