import sentencepiece as sp
import sys
import csv

# follow fairseq's definition
UNK_TOKEN, UNK_TOKEN_ID = "<unk>", 3
BOS_TOKEN, BOS_TOKEN_ID = "<s>", 0
EOS_TOKEN, EOS_TOKEN_ID = "</s>", 2
PAD_TOKEN, PAD_TOKEN_ID = "<pad>", 1


if __name__ == '__main__':
    
    data_dir = '/home/sean/battleship/s3prl/data/test'
    src_lang = 'en'
    tgt_lang = 'de'
    split = 'train'
    file_path = f'{data_dir}/{split}_st_{src_lang}_{tgt_lang}.tsv'
    key = 'tgt_text'
    tmp_file = 'tmp.txt'

    output_path_prefix = f'{data_dir}/spm_{src_lang}_{tgt_lang}_{key}'

    with open(tmp_file, 'w') as f_out:
        with open(file_path, 'r') as f_in:
            reader = csv.DictReader(f_in, delimiter='\t')
            for line in reader:
                print(line[key], file=f_out)


    sp.SentencePieceTrainer.train(
        input=[tmp_file], 
        model_prefix=output_path_prefix,
        model_type='char',
        character_coverage=1,
        unk_id=f"{UNK_TOKEN_ID}",
        bos_id=f"{BOS_TOKEN_ID}",
        eos_id=f"{EOS_TOKEN_ID}",
        pad_id=f"{PAD_TOKEN_ID}",
    )


    # expose fiarseq dictionary
    spm = sp.SentencePieceProcessor()
    spm.Load(output_path_prefix + ".model")
    vocab = {i: spm.IdToPiece(i) for i in range(spm.GetPieceSize())}
    assert (
        vocab.get(UNK_TOKEN_ID) == UNK_TOKEN
        and vocab.get(PAD_TOKEN_ID) == PAD_TOKEN
        and vocab.get(BOS_TOKEN_ID) == BOS_TOKEN
        and vocab.get(EOS_TOKEN_ID) == EOS_TOKEN
    )
    vocab = {
        i: s
        for i, s in vocab.items()
        if s not in {UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN}
    }
    with open(output_path_prefix + ".txt", "w") as f_out:
        for _, s in sorted(vocab.items(), key=lambda x: x[0]):
            f_out.write(f"{s} 1\n")