#!/usr/bin/env bash
# *********************************************************************************************
#   FileName     [ decode.sh ]
#   Synopsis     [ neural vocoder decoding & objective evaluation script for voice conversion ]
#   Author       [ Wen-Chin Huang (https://github.com/unilight) ]
#   Copyright    [ Copyright(c), Toda Lab, Nagoya University, Japan ]
# *********************************************************************************************

upstream=$1
trgspk=$2
tag=$3
ep=$4
voc_dir=$5
list_file=$6

# check arguments
if [ $# != 6 ]; then
    echo "Usage: $0 <upstream> <trgspk> <tag> <ep> <voc_dir> <list_file>"
    exit 1
fi

expname=a2o_vc_vcc2020_${tag}_${trgspk}_${upstream}
expdir=result/downstream/${expname}
outdir=${expdir}/custom_test
eval_list_file=$(realpath ${list_file})

voc_name=$(basename ${voc_dir} | cut -d"_" -f 1)
voc_checkpoint="$(find "${voc_dir}" -name "*.pkl" -print0 | xargs -0 ls -t | head -n 1)"
voc_conf="$(find "${voc_dir}" -name "config.yml" -print0 | xargs -0 ls -t | head -n 1)"
voc_stats="$(find "${voc_dir}" -name "stats.h5" -print0 | xargs -0 ls -t | head -n 1)"
wav_dir=${outdir}/${voc_name}_wav
hdf5_norm_dir=${outdir}/hdf5_norm
rm -rf ${wav_dir}; mkdir -p ${wav_dir}
rm -rf ${hdf5_norm_dir}; mkdir -p ${hdf5_norm_dir}

python run_downstream.py -m evaluate -t custom_test -d a2o-vc-vcc2020 \
    -n ${expname} \
    -u ${upstream} \
    -e ${expdir}/states-${ep}.ckpt \
    -o "config.downstream_expert.datarc.eval_list_file=${eval_list_file},,config.downstream_expert.trgspk=${trgspk}"

# normalize and dump them
echo "Normalizing..."
parallel-wavegan-normalize \
    --skip-wav-copy \
    --config "${voc_conf}" \
    --stats "${voc_stats}" \
    --rootdir "${outdir}" \
    --dumpdir ${hdf5_norm_dir} \
    --verbose 1
echo "successfully finished normalization."

# decoding
echo "Decoding start."
parallel-wavegan-decode \
    --dumpdir ${hdf5_norm_dir} \
    --checkpoint "${voc_checkpoint}" \
    --outdir ${wav_dir} \
    --verbose 1
echo "successfully finished decoding."