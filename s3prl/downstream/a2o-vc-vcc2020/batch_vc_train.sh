#!/usr/bin/env bash
# *********************************************************************************************
#   FileName     [ batch_vc_train.sh ]
#   Synopsis     [ Script to train any-to-one voice conversion models in batch mode ]
#   Author       [ Wen-Chin Huang (https://github.com/unilight) ]
#   Copyright    [ Copyright(c), Toda Lab, Nagoya University, Japan ]
# *********************************************************************************************

upstream=$1
config=$2
tag=$3
part=$4

set -e

# check arguments
if [ $# != 4 ]; then
    echo "Usage: $0 <upstream> <config> <tag> <part>"
    exit 1
fi

if [ ${part} == "task1_female" ]; then
    trgspks=("TEF1" "TEF2")
elif [ ${part} == "task1_male" ]; then
    trgspks=("TEM1" "TEM2")
elif [ ${part} == "task1_all" ]; then
    trgspks=("TEF1" "TEF2" "TEM1" "TEM2")
elif [ ${part} == "task2_fin" ]; then
    trgspks=("TFF1" "TFM1")
elif [ ${part} == "task2_ger" ]; then
    trgspks=("TGF1" "TGM1")
elif [ ${part} == "task2_man" ]; then
    trgspks=("TMF1" "TMM1")
else
    echo "Invalid part specification  Please specify from the following choices:"
    echo "task1_female, task1_male, task1_all, task2_fin, task2_ger, task2_man"
    exit 1
fi

echo "Script starting time: $(date +%T)"

pids=() # initialize pids
for trgspk in "${trgspks[@]}"; do
(
    expname=a2o_vc_vcc2020_${tag}_${trgspk}_${upstream}
    expdir=result/downstream/${expname}
    mkdir -p ${expdir}
    python run_downstream.py -m train \
        --config ${config} \
        -n ${expname} \
        -u ${upstream} \
        -d a2o-vc-vcc2020 \
        -o "config.downstream_expert.trgspk='${trgspk}'" \
        > ${expdir}/train.log 2>&1
) &
pids+=($!) # store background pids
done
i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
[ ${i} -gt 0 ] && echo "$0: ${i} background jobs failed." && false
