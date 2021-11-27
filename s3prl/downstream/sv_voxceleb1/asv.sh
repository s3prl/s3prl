#!/bin/bash

set -e

# required
default_lrs="1e-4"

# required
metric_higher_better=false

# optional
default_explore_ratio=0.00001 # Don't need explore, 1e-4 always the best

# required
function get_eval_result() {
    if [ "$#" != "2" ]; then
        echo "Usage: get_eval_result EXPDIR dev|test"
        exit 2
    fi

    local expdir=$1
    local mode=$2

    report=$expdir/report.txt
    if [ -f "$report" ]; then
        best_eer=$(cat $report | tail -n 1 | cut -d " " -f 2)
        if [[ "$best_eer" =~ ^[0-9]+\.[0-9]+$ ]]; then
            echo "eer ${best_eer}"
        fi
    fi
}

# required
function single_trial() {
    if [ "$#" != "4" ]; then
        echo "Usage: single_trial EXPDIR UPSTREAM OVERRIDE RUN_TEST"
        exit 2
    fi

    local expdir=$1
    local upstream=$2
    local override=$3
    local run_test=$4

    local test_result="$(get_eval_result $expdir "test")"
    if [ -z "$test_result" ]; then
        python3 run_downstream.py -m train -a -u $upstream -d sv_voxceleb1 -p $expdir -o $override
        voxceleb1=""  # use the same path saved in the checkpoint
        if [ $run_test = true ]; then
            # full run use pre-defined checkpoints to evaluate, all checkpoints will be too many
            ckpt_names=""
        else
            # partial run use all checkpoints to evaluate
            ckpt_names="$(ls -rt $expdir | grep "ckpt")"
        fi
        ./downstream/sv_voxceleb1/test_expdir.sh "$expdir" "$voxceleb1" $ckpt_names
    else
        echo "Test result is find:"
        echo "$test_result"
        echo "Skip train and test..."
    fi
}
