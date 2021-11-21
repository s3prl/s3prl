#!/bin/bash

set -e

# required
default_lrs="1e-3 1e-4 1e-5"

# required
metric_higher_better=true

# optional
default_explore_ratio=0.05

# required
function get_eval_result() {
    if [ "$#" != "2" ]; then
        echo "Usage: get_eval_result EXPDIR dev|test"
        exit 2
    fi

    local expdir=$1
    local mode=$2
    local result_file=$expdir/$mode.result

    if [ -f "$result_file" ]; then
        local metric="slot_type_f1"
        if [[ "$(grep -E " $metric" $result_file)" =~ ([0-9]+\.[0-9]+) ]]; then
            echo -n "$metric ${BASH_REMATCH[1]} "
        fi

        local metric="slot_value_cer"
        if [[ "$(grep -E " $metric" $result_file)" =~ ([0-9]+\.[0-9]+) ]]; then
            echo -n "$metric ${BASH_REMATCH[1]} "
        fi

        echo
    fi
}

# optional
function eval_best_dev() {
    if [ "$#" != "2" ]; then
        echo "Usage: find_eval_ckpt EXPDIR dev|test"
        exit 2
    fi

    local expdir=$1
    local mode=$2
    local eval_ckpt=$(ls -t $expdir | grep -E ".*dev.*\.ckpt" | head -n 1)  # take the best checkpoint on dev
    if [ -z "$eval_ckpt" ]; then
        echo "The best development checkpoint not found. Deganerate to use the last checkpoint"
        echo "This should not happen during the full benchmarking"
        local eval_ckpt=$(ls -t $expdir | grep -E ".*\.ckpt" | head -n 1)
    fi
    eval_ckpt=$expdir/$eval_ckpt

    if [ "$mode" == "dev" ]; then
        local split="dev"
    elif [ "$mode" == "test" ]; then
        local split="test"
    else
        echo Invalid mode argument: $mode
        exit 1
    fi

    echo "$mode with $eval_ckpt..."
    python3 run_downstream.py -m evaluate -e $eval_ckpt -t $split > $expdir/${mode}.result
}

# required
function single_trial() {
    if [ "$#" != "6" ]; then
        echo "Usage: single_trial EXPDIR UPSTREAM LR OPTIMIZE_RATIO OVERRIDE RUN_TEST"
        exit 2
    fi

    local expdir=$1
    local upstream=$2
    local lr=$3
    local optimize_ratio=$4
    local override=$5
    local run_test=$6

    if [ ! -z $override ]; then
        override=",,$override"
    fi

    local dev_result="$(get_eval_result $expdir "dev")"
    if [ -z "$dev_result" ]; then
        python3 run_downstream.py -m train -a -u $upstream -d ctc -c downstream/ctc/snips.yaml -p $expdir \
            -o config.optimizer.lr=$lr,,config.runner.optimize_ratio=${optimize_ratio}${override}
        
        eval_best_dev $expdir "dev"
    else
        echo "Dev result is find:"
        echo "$dev_result"
        echo "Skip train and dev..."
    fi

    if [ $run_test = true ]; then
        local test_result="$(get_eval_result $expdir "test")"
        if [ -z "$test_result" ]; then
            eval_best_dev $expdir "test"
        else
            echo "Test result is find:"
            echo "$test_result"
            echo "Skip test..."
        fi
    fi
}
