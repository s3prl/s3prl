import sentencepiece as spm
import sys

if __name__ == '__main__':
    
    vocab_size = int(sys.argv[1])
    model_prefix = sys.argv[2]
    files = sys.argv[3:]
    pad_id = 0
    bos_id = 1
    eos_id = 2
    unk_id = 3

    spm.SentencePieceTrainer.train(
        input=files, 
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
        unk_id=unk_id,
    )