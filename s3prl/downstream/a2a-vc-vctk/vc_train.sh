#!/usr/bin/env bash
# *********************************************************************************************
#   FileName     [ vc_train.sh ]
#   Synopsis     [ Script to train an any-to-any voice conversion model ]
#   Author       [ Wen-Chin Huang (https://github.com/unilight) ]
#   Copyright    [ Copyright(c), Toda Lab, Nagoya University, Japan ]
# *********************************************************************************************

upstream=$1
config=$2
tag=$3

set -e

# check arguments
if [ $# != 3 ]; then
    echo "Usage: $0 <upstream> <config> <tag>"
    exit 1
fi

echo "Training script starting time: $(date +%T)"

expname=a2a_vc_vctk_${tag}_${upstream}
expdir=result/downstream/${expname}
mkdir -p ${expdir}

python run_downstream.py -m train \
    --config ${config} \
    -p ${expdir} \
    -u ${upstream} \
    -d a2a-vc-vctk \
    > ${expdir}/train.log 2>&1 || echo "Error. Log file can be found in ${expdir}/train.log"
