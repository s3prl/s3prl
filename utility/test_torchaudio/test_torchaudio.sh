#!/bin/bash

set -e

file=$1

source ~/miniconda3/etc/profile.d/conda.sh

conda activate benchmark
pip show torchaudio
python3 load_audio.py $file 1
conda deactivate

conda activate audio
pip show torchaudio
python3 load_audio.py $file 2
conda deactivate

python3 test_torchallclose.py 1.pth 2.pth

