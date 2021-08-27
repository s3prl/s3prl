import csv
import argparse
import tempfile
import sentencepiece as sp
import shutil

# fairseq's special token
UNK_TOKEN, UNK_TOKEN_ID = "<unk>", 3
BOS_TOKEN, BOS_TOKEN_ID = "<s>", 0
EOS_TOKEN, EOS_TOKEN_ID = "</s>", 2
PAD_TOKEN, PAD_TOKEN_ID = "<pad>", 1

def create_sentencepiece(filenames, model_type, vocab_size, output_prefix):

    sp.SentencePieceTrainer.train(
        input=','.join(filenames),
        model_prefix=output_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=1.0,
        unk_id=UNK_TOKEN_ID,
        bos_id=BOS_TOKEN_ID,
        eos_id=EOS_TOKEN_ID,
        pad_id=PAD_TOKEN_ID,
    )

    spm = sp.SentencePieceProcessor(
        model_file=f'{output_prefix}.model'
    )

    vocab = {i: spm.IdToPiece(i) for i in range(spm.GetPieceSize())}

    assert vocab.get(UNK_TOKEN_ID) == UNK_TOKEN
    assert vocab.get(BOS_TOKEN_ID) == BOS_TOKEN
    assert vocab.get(EOS_TOKEN_ID) == EOS_TOKEN
    assert vocab.get(PAD_TOKEN_ID) == PAD_TOKEN

    vocab = {
        i: s for i, s in vocab.items()
        if s not in {UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN}
    }

    with open(f'{output_prefix}.txt', 'w') as f:
        for _, s in sorted(vocab.items(), key=lambda x: x[0]):
            print(f'{s} 1', file=f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input_tsv')
    parser.add_argument('-s', '--src-key', default='src_text')
    parser.add_argument('-t', '--tgt-key', default='tgt_text')
    parser.add_argument('-c', '--combine', action='store_true')
    parser.add_argument('-o', '--output-dir')
    parser.add_argument('-n', '--vocab-size', default=1000)
    parser.add_argument('--model-type', default='char')
    args = parser.parse_args()

    with tempfile.NamedTemporaryFile(mode='w') as src_f:
        with tempfile.NamedTemporaryFile(mode='w') as tgt_f:
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
                    print(line[args.src_key], file=src_f)
                    print(line[args.tgt_key], file=tgt_f)
            if not args.combine:
                create_sentencepiece(
                    [src_f.name],
                    args.model_type,
                    args.vocab_size,
                    f'{args.output_dir}/spm-{args.src_key}',
                )
                create_sentencepiece(
                    [tgt_f.name],
                    args.model_type,
                    args.vocab_size,
                    f'{args.output_dir}/spm-{args.tgt_key}',
                )
            else:
                create_sentencepiece(
                    [src_f.name, tgt_f.name],
                    args.model_type,
                    args.vocab_size,
                    f'{args.output_dir}/spm-{args.src_key}',
                )
                for s in ['model', 'vocab', 'txt']:
                    shutil.copyfile(
                        f'{args.output_dir}/spm-{args.src_key}.{s}',
                        f'{args.output_dir}/spm-{args.tgt_key}.{s}',
                    )
