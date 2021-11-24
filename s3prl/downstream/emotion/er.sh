#!/bin/bash

set -e

# required
default_lrs="1e-0 1e-1 1e-2 1e-3 1e-4 1e-5"

# required
metric_higher_better=true

# optional
default_explore_ratio=0.2

# required
function get_eval_result() {
    if [ "$#" != "2" ]; then
        echo "Usage: get_eval_result FOLDS_EXPDIR dev|test"
        exit 2
    fi

    local folds_expdir=$1
    local mode=$2

    result=""
    for expdir in $(ls -d $folds_expdir/*/ | grep fold);
    do
        single_fold_result=$(get_eval_result_single_fold $expdir $mode | cut -d " " -f 2)
        result="$single_fold_result $result"
    done
    avg_acc=$(python3 utility/mean.py $result)
    echo "acc $avg_acc"
}

function get_eval_result_single_fold() {
    if [ "$#" != "2" ]; then
        echo "Usage: get_eval_result_single_fold EXPDIR dev|test"
        exit 2
    fi

    local expdir=$1
    local mode=$2
    local result_file=$expdir/$mode.result

    if [ -f "$result_file" ]; then
        local metric="acc"
        if [[ "$(grep -E " $metric" $result_file)" =~ ([0-9]+\.[0-9]+) ]]; then
            echo "$metric ${BASH_REMATCH[1]}"
        fi
    fi
}

# optional
function eval_best_dev() {
    if [ "$#" != "2" ]; then
        echo "Usage: eval_best_dev EXPDIR dev|test"
        exit 2
    fi

    local expdir=$1
    local mode=$2
    local eval_ckpt=$(ls -t $expdir | grep -E ".*dev.*\.ckpt" | head -n 1)  # take the best checkpoint on dev
    if [ -z "$eval_ckpt" ]; then
        echo "The best development checkpoint not found. Deganerate to use the last checkpoint"
        echo "This should not happen during the full benchmarking"
        eval_ckpt=$(ls -t $expdir | grep -E ".*\.ckpt" | head -n 1)
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
    if [ "$#" != "4" ]; then
        echo "Usage: single_trial FOLDS_EXPDIR UPSTREAM OVERRIDE RUN_TEST"
        exit 2
    fi

    local folds_expdir=$1
    local upstream=$2
    local override=$3
    local run_test=$4

    for fold in fold1 fold2 fold3 fold4 fold5;
    do
        local expdir=$folds_expdir/$fold
        local dev_result="$(get_eval_result_single_fold $expdir "dev")"
        if [ -z "$dev_result" ]; then
            python3 run_downstream.py -m train -a -u $upstream -d emotion -p $expdir \
                -o $override,,config.downstream_expert.datarc.test_fold=$fold
            eval_best_dev $expdir "dev"
        else
            echo "Dev result is find:"
            echo "$dev_result"
            echo "Skip train and dev..."
        fi

        if [ $run_test = true ]; then
            local test_result="$(get_eval_result_single_fold $expdir "test")"
            if [ -z "$test_result" ]; then
                eval_best_dev $expdir "test"
            else
                echo "Test result is find:"
                echo "$test_result"
                echo "Skip test..."
            fi
        fi
    done
}
