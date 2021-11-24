#!/usr/bin/env bash
# *********************************************************************************************
#   FileName     [ decode.sh ]
#   Synopsis     [ neural vocoder decoding & objective evaluation script for voice conversion ]
#   Author       [ Wen-Chin Huang (https://github.com/unilight) ]
#   Copyright    [ Copyright(c), Toda Lab, Nagoya University, Japan ]
# *********************************************************************************************

voc_dir=$1
outdir=$2
trgspk=$3

# check arguments
if [ $# != 3 ]; then
    echo "Usage: $0 <voc_dir> <outdir> <trgspk>"
    exit 1
fi

voc_name=$(basename ${voc_dir} | cut -d"_" -f 1)
voc_checkpoint="$(find "${voc_dir}" -name "*.pkl" -print0 | xargs -0 ls -t | head -n 1)"
voc_conf="$(find "${voc_dir}" -name "config.yml" -print0 | xargs -0 ls -t | head -n 1)"
voc_stats="$(find "${voc_dir}" -name "stats.h5" -print0 | xargs -0 ls -t | head -n 1)"
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
python downstream/a2o-vc-vcc2020/evaluate.py \
    --wavdir ${wav_dir} \
    --data_root downstream/a2o-vc-vcc2020/data \
    --trgspk ${trgspk}
