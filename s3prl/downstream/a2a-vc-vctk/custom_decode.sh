#!/bin/bash
# *********************************************************************************************
#   FileName     [ decode.sh ]
#   Synopsis     [ PWG decoding & objective evaluation script for voice conversion ]
#   Author       [ Wen-Chin Huang (https://github.com/unilight) ]
#   Copyright    [ Copyright(c), Toda Lab, Nagoya University, Japan ]
# *********************************************************************************************

tag=$1
upstream=$2
ep=$3
voc_expdir=$4
list_file=$5

# check arguments
if [ $# != 5 ]; then
    echo "Usage: $0 <tag> <upstream> <ep< <voc_expdir> <list>"
    exit 1
fi

expname=a2a_vc_vctk_${tag}_${upstream}
outdir=result/downstream/${expname}/custom_test
eval_pair_list_file=$(realpath ${list_file})

voc_name=$(basename ${voc_expdir} | cut -d"_" -f 1)

voc_checkpoint="$(find "${voc_expdir}" -name "*.pkl" -print0 | xargs -0 ls -t | head -n 1)"
voc_conf="$(find "${voc_expdir}" -name "config.yml" -print0 | xargs -0 ls -t | head -n 1)"
voc_stats="$(find "${voc_expdir}" -name "stats.h5" -print0 | xargs -0 ls -t | head -n 1)"
wav_dir=${outdir}/${voc_name}_wav
hdf5_norm_dir=${outdir}/hdf5_norm
rm -rf ${wav_dir}; mkdir -p ${wav_dir}
rm -rf ${hdf5_norm_dir}; mkdir -p ${hdf5_norm_dir}

# run inference
python run_downstream.py -m evaluate -t custom_test -d a2a-vc-vctk \
    -n ${expname} \
    -u ${upstream} \
    -e result/downstream/${expname}/states-${ep}.ckpt \
    -o "config.downstream_expert.datarc.eval_pair_list_file=${eval_pair_list_file}"

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