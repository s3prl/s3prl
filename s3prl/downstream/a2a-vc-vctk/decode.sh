#!/bin/bash
# *********************************************************************************************
#   FileName     [ decode.sh ]
#   Synopsis     [ PWG decoding & objective evaluation script for voice conversion ]
#   Author       [ Wen-Chin Huang (https://github.com/unilight) ]
#   Copyright    [ Copyright(c), Toda Lab, Nagoya University, Japan ]
# *********************************************************************************************

voc_expdir=$1
outdir=$2

# check arguments
if [ $# != 2 ]; then
    echo "Usage: $0 <voc_expdir> <outdir>"
    exit 1
fi

voc_name=$(basename ${voc_expdir} | cut -d"_" -f 1)

voc_checkpoint="$(find "${voc_expdir}" -name "*.pkl" -print0 | xargs -0 ls -t | head -n 1)"
voc_conf="$(find "${voc_expdir}" -name "config.yml" -print0 | xargs -0 ls -t | head -n 1)"
voc_stats="$(find "${voc_expdir}" -name "stats.h5" -print0 | xargs -0 ls -t | head -n 1)"
wav_dir=${outdir}/${voc_name}_wav
hdf5_norm_dir=${outdir}/hdf5_norm
rm -rf ${wav_dir}; mkdir -p ${wav_dir}
rm -rf ${hdf5_norm_dir}; mkdir -p ${hdf5_norm_dir}

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

# evaluation
echo "Evaluation start."
for num in 10; do
    python downstream/a2a-vc-vctk/evaluate.py \
        --wavdir ${wav_dir} \
        --samples ${num} \
        --task task1 \
        --data_root ./downstream/a2o-vc-vcc2020/data 
done
