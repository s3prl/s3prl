#!/bin/bash

set -e

# required
default_lrs="1e-4"

# required
metric_higher_better=false

# optional
default_explore_ratio=1
stage1=false
stage2=true

# required
function get_eval_result() {
    if [ "$#" != "2" ]; then
        echo "Usage: get_eval_result SPKS_EXPDIR dev|test"
        exit 2
    fi

    local spks_expdir=$1
    local mode=$2
    if [ "$mode" == "dev" ]; then
        echo "Only support test mode" && exit 1
    fi

    result_file="${spks_expdir}/summary"
    if [ -f "$result_file" ]; then
        if [[ "$(cat $result_file | tail -n 1)" =~ :(-?[0-9]+\.[0-9]+)[[:space:]](-?[0-9]+\.[0-9]+)[[:space:]](-?[0-9]+\.[0-9]+)[[:space:]](-?[0-9]+\.[0-9]+)[[:space:]](-?[0-9]+\.[0-9]+)[[:space:]](-?[0-9]+\.[0-9]+)[[:space:]](-?[0-9]+\.[0-9]+) ]]; then
            local mcd=${BASH_REMATCH[1]}
            local wer=${BASH_REMATCH[6]}
            local asv=${BASH_REMATCH[7]}
            echo "mcd ${mcd} wer ${wer} asv ${asv}"
        fi
    fi
}

# required
function single_trial() {
    if [ "$#" != "4" ]; then
        echo "Usage: single_trial SPKS_EXPDIR UPSTREAM OVERRIDE RUN_TEST"
        exit 2
    fi

    # install the additional packages dedicated for VC
    pip install -r downstream/a2o-vc-vcc2020/requirements.txt

    local spks_expdir=$1
    local upstream=$2
    local override=$3
    local run_test=$4

    local test_result="$(get_eval_result $spks_expdir "test")"
    if [ ! -z "$test_result" ]; then
        echo "Test result is find:"
        echo "$test_result"
        echo "Skip test..."
    else
        for spk in TEF1 TEF2 TEM1 TEM2;
        do
            ./downstream/a2o-vc-vcc2020/batch_vc_train.sh $upstream \
                ./downstream/a2o-vc-vcc2020/config.yaml superb $spk $spks_expdir $override
        done
        ./downstream/a2o-vc-vcc2020/batch_vc_decode.sh $upstream \
            task1 superb ./downstream/a2o-vc-vcc2020/hifigan_vctk+vcc2020/ $spks_expdir \
            > $spks_expdir/summary
    fi
}
