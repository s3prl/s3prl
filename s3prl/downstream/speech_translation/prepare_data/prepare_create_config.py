import argparse
import yaml
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--sp-model', required=True)
    parser.add_argument('-v', '--vocab-file', required=True)
    parser.add_argument('-a', '--audio-dir', required=True)
    parser.add_argument('-o', '--output', required=True)
    args = parser.parse_args()

    print(args)

    config = {
        'bpe_tokenizer': {
            'bpe': 'sentencepiece',
            'sentencepiece_model': os.path.abspath(args.sp_model),
        },
        'vocab_filename': args.vocab_file,
        'audio_root': args.audio_dir,
        'use_audio_input': True,
    }

    with open(args.output, 'w') as f:
        yaml.dump(config, f)
