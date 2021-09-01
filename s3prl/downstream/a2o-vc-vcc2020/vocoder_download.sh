#!/usr/bin/env bash
# *********************************************************************************************
#   FileName     [ vocoder_download.sh ]
#   Synopsis     [ Script to download pretrained neural vocoders ]
#   Author       [ Wen-Chin Huang (https://github.com/unilight) ]
#   Copyright    [ Copyright(c), Toda Lab, Nagoya University, Japan ]
# *********************************************************************************************

# This script is based on the following links:
# https://raw.githubusercontent.com/espnet/espnet/master/egs/vcc20/vc1_task1/local/pretrained_model_download.sh
# https://github.com/espnet/espnet/blob/master/utils/download_from_google_drive.sh

download_dir=$1

# check arguments
if [ $# != 1 ]; then
    echo "Usage: $0 <download_dir>"
    exit 1
fi

pwg_task1_url="https://drive.google.com/open?id=11KKux-du6fvsMMB4jNk9YH23YUJjRcDV"
pwg_task2_url="https://drive.google.com/open?id=1li9DLZGnAheWZrB4oXGo0KWq-fHuFH_l"

download_from_google_drive() {
    share_url=$1
    dir=$2
    file_ext=$3

    # make temp dir
    [ ! -e "${dir}" ] && mkdir -p "${dir}"
    tmp=$(mktemp "${dir}/XXXXXX.${file_ext}")

    # download & decompress
    file_id=$(echo "${share_url}" | cut -d"=" -f 2)
    gdown --id "${file_id}" -O "${tmp}"
    tar xvzf "${tmp}" -C "${dir}"

    # remove tmp
    rm "${tmp}"
}

download_from_google_drive ${pwg_task1_url} ${download_dir}/pwg_task1 ".tar.gz"
download_from_google_drive ${pwg_task2_url} ${download_dir}/pwg_task2 ".tar.gz"
echo "Successfully finished donwload of pretrained models."
