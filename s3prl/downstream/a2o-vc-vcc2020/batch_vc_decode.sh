#!/usr/bin/env bash
# *********************************************************************************************
#   FileName     [ batch_vc_decode.sh ]
#   Synopsis     [ Script to perform decoding for any-to-one voice conversion models in batch mode ]
#   Author       [ Wen-Chin Huang (https://github.com/unilight) ]
#   Copyright    [ Copyright(c), Toda Lab, Nagoya University, Japan ]
# *********************************************************************************************

upstream=$1
task=$2
tag=$3

set -e

# check arguments
if [ $# != 3 ]; then
    echo "Usage: $0 <upstream> <task> <tag>"
    exit 1
fi

start_ep=4000
interval=1000
end_ep=10000

if [ ${task} == "task1" ]; then
    trgspks=("TEF1" "TEF2" "TEM1" "TEM2")
elif [ ${task} == "task2" ]; then
    trgspks=("TFF1" "TFM1" "TGF1" "TGM1" "TMF1" "TMM1")
fi

for trgspk in "${trgspks[@]}"; do
    for ep in $(seq ${start_ep} ${interval} ${end_ep}); do
        echo "Objective evaluation: Ep ${ep}; trgspk ${trgspk}"
        expname=a2o_vc_vcc2020_${tag}_${trgspk}_${upstream}
        expdir=../../result/downstream/${expname}
        ./decode.sh pwg_${task}/ ${expdir}/${ep} ${trgspk}
    done
done

python find_best_epoch.py --upstream ${upstream} --tag ${tag} --task ${task}
