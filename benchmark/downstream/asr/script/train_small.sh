#!/bin/bash

# $1 : experiment name
# $2 : cuda id

CONFIG="small"

DIR="/home/leo/d/Benchmarking/S3PRL/benchmark/downstream/asr"

echo "Start running training process of E2E ASR"
CUDA_VISIBLE_DEVICES=$2 python3 ${DIR}/main.py --config ${DIR}/config/${CONFIG}.yaml \
    --name $1 \
    --njobs 16
    --seed 0 \
    --logdir ${DIR}/log/ \
    --ckpdir ${DIR}/ckpt/ \
    --outdir ${DIR}/result/ \
    --reserve_gpu 0 \

