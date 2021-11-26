#!/bin/bash
# *********************************************************************************************
#   FileName     [ vctk_download.sh ]
#   Synopsis     [ Script to download the VCTK dataset ]
#   Reference    [ https://github.com/kan-bayashi/ParallelWaveGAN/blob/master/egs/vctk/voc1/local/data_download.sh ]
# *********************************************************************************************

# Copyright 2020 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

download_dir=$1

# check arguments
if [ $# != 1 ]; then
    echo "Usage: $0 <download_dir>"
    exit 1
fi

set -euo pipefail

cwd=$(pwd)
if [ ! -e "${download_dir}/VCTK-Corpus" ]; then
    mkdir -p "${download_dir}"
    cd "${download_dir}" || exit 1;
    wget http://www.udialogue.org/download/VCTK-Corpus.tar.gz
    tar xvzf ./VCTK-Corpus.tar.gz
    rm ./VCTK-Corpus.tar.gz
    cd "${cwd}" || exit 1;
    echo "Successfully downloaded data."
else
    echo "Already exists. Skipped."
fi
