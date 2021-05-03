#!/bin/bash
set -e
set -x

if [ $# -ne 1 ]; then
    echo "Usage: $0 <corpora dir>"
    echo "eg: $0 /all/my/datasets/"
    exit 1
fi

script_dir=$(dirname $0)
script_dir=$(readlink -f $script_dir)
corpora_root=$1
cd $corpora_root

if [ -s SNIPS/all.iob.snips.txt ];then
    echo 'Preprocessed text file exist, skip!'
else
    if [ ! -d aws-lex-noisy-spoken-language-understanding ];then
        echo 'Start downloading text files...'
        git clone https://github.com/aws-samples/aws-lex-noisy-spoken-language-understanding.git
    fi

    echo 'Start preparing text files...'
    mkdir -p SNIPS
    python3 "$script_dir/snips_text_norm.py"
    python3 "$script_dir/snips_preprocess.py" text aws-lex-noisy-spoken-language-understanding SNIPS
    rm SNIPS/single*
fi

if [ -s SNIPS/valid/Salli-snips-valid-168.wav ];then
    echo 'Preprocessed audio file exist, skip!'
else
    if [ ! -d audio_slu ];then
        echo 'Start downloading audio files...'
        wget https://shangwel-asr-evaluation.s3-us-west-2.amazonaws.com/audio_slu_v3.zip
        echo 'Start unzipping audio files...'
        unzip audio_slu_v3.zip > tmp.log
    fi

    echo 'Start converting audio files...'
    python "$script_dir/snips_preprocess.py" audio audio_slu SNIPS
fi

