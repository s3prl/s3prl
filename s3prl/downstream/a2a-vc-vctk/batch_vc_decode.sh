#!/bin/bash

task="task1"

upstream=$1
tag=$2
vocoder=$3

set -e

# check arguments
if [ $# != 3 ]; then
    echo "Usage: $0 <upstream> <task> <tag> <vocoder>"
    exit 1
fi

start_ep=20000
#start_ep=10000
interval=2000
end_ep=50000

if [ ${task} == "task1" ]; then
    trgspks=("TEF1" "TEF2" "TEM1" "TEM2")
elif [ ${task} == "task2" ]; then
    trgspks=("TFF1" "TFM1" "TGF1" "TGM1" "TMF1" "TMM1")
fi

for ep in $(seq ${start_ep} ${interval} ${end_ep}); do
    echo "Objective evaluation: Ep ${ep}"
    expname=a2a_vc_vctk_${tag}_${upstream}
    #expname=test_a2a_vc_vctk_${upstream}_${tag}
    expdir=../../result/downstream/${expname}
    ./decode.sh ${vocoder} ${expdir}/${ep} #> /dev/null 2>&1
    # grep 'Mean' ${expdir}/${ep}/pwg_wav/obj.log
done

voc_name=$(basename ${vocoder} | cut -d"_" -f 1)

python find_best_epoch.py \
    --start_epoch ${start_ep} \
    --end_epoch ${end_ep} \
    --step_epoch ${interval} \
    --upstream ${upstream} --tag ${tag} --task ${task} --vocoder ${vocoder}
