#!/bin/bash

set -e

if [ $# -lt "3" ]; then
    echo "$0 [upstream] [exps_root] [override] ([lr1: 1e-0] [lr2: 1e-1] ...)"
    echo "The expdirs will be located at exps_root/upstream/override/1e-0, exps_root/upstream/override/1e-1"
    echo "[override] can be used to override any default fields in the args or the config file"
    echo "Please set [override] to 'none' if you wish to benchmark with the default setting"
    echo "eg. $0 wav2vec2 result/task none"
    echo "You can override the upstream_layer_selection for layer-wise benchmarking"
    echo "eg. $0 wav2vec2 result/task args.upstream_layer_selection=3"
    exit 1
fi

upstream=$1
exps_root=$2
override=$3
shift 3

if [ $# == "0" ]; then
    lrs="1e-0 1e-1 1e-2 1e-3 1e-4 1e-5 1e-6"
else
    lrs="$*"
fi

# constant
upstream_dir=$exps_root/$upstream/override_$override

function get_eval_result() {
    result_file=$1
    if [ -f $result_file ]; then
        metric="per"
        if [[ "$(grep -E " $metric" $result_file)" =~ ([0-9]+.[0-9]+) ]]; then
            echo "$metric ${BASH_REMATCH[1]}"
        fi
    fi
}

function single_trial() {
    optimize_ratio=$1
    lr=$2
    run_test=$3

    expdir=$upstream_dir/optimize_ratio_$optimize_ratio/lr_$lr
    mkdir -p $expdir
    echo "Try learning $lr... results will be saved at $expdir"

    if [ -z "$(get_eval_result $expdir/dev.result)" ]; then
        echo "Train with $optimize_ratio of the all optimization steps..."
        python3 run_downstream.py -m train -a -u $upstream -d ctc -c downstream/ctc/libriphone.yaml -p $expdir \
            -o config.runner.optimize_ratio=$optimize_ratio,,config.optimizer.lr=$lr

        eval_ckpt=$(ls -t $expdir | grep -E ".*dev.*\.ckpt" | head -n 1)  # take the best checkpoint on dev
        if [ -z $eval_ckpt ]; then
            echo "The best development checkpoint not found. Deganerate to use the last checkpoint"
            echo "This should not happen during the full benchmarking"
            eval_ckpt=$(ls -t $expdir | grep -E ".*\.ckpt" | head -n 1)
        fi
        eval_ckpt=$expdir/$eval_ckpt

        echo "Dev with $eval_ckpt..."
        python3 run_downstream.py -m evaluate -e $eval_ckpt -t "dev" > $expdir/dev.result
    else
        echo "Dev result is find, skip train and dev..."
    fi

    if [ $run_test = true ]; then
        if [ -z "$(get_eval_result $expdir/test.result)" ]; then
            echo "Test with $eval_ckpt..."
            python3 run_downstream.py -m evaluate -e $eval_ckpt -t "test" > $expdir/test.result
        else
            echo "Test result is find, skip test..."
        fi
    fi
}

echo "Exploring learning rate... $lrs"
optimize_ratio=0.05
for lr in $lrs;
do
    single_trial $optimize_ratio $lr false
done
explore_dir=$upstream_dir/optimize_ratio_$optimize_ratio

echo "Report exploration result..."
explore_summary=$explore_dir/summary
[ -f $explore_summary ] && rm $explore_summary
for expdir in $(ls -d $explore_dir/*);
do
    echo $(basename $expdir): $(get_eval_result $expdir/dev.result) >> $explore_summary
done

echo "Picking the best learning rate..."
best_dev=$(cat $explore_dir/summary | sort -gk 3 | head -n 1)
[[ "$best_dev" =~ lr_(.*): ]] && best_lr=${BASH_REMATCH[1]}
if [ -z $best_lr ]; then
    echo "The training/dev/test during the learning rate exploration were not completed or files are corrupted"
    echo "Please delete $upstream_dir and try again"
    exit 1
fi
echo "Best learning rate: $best_lr"

echo "Final full training..."
single_trial 1 $best_lr true
full_dir=$upstream_dir/optimize_ratio_1

echo "Report full training result..."
full_summary=$full_dir/summary
[ -f $full_summary ] && rm $full_summary
for expdir in $(ls -d $full_dir/*);
do
    echo $(basename $expdir): $(get_eval_result $expdir/test.result) >> $full_summary
done

summary=$upstream_dir/summary
[ -f $summary ] && rm $summary
echo Explore learning rate >> $summary
cat $explore_summary >> $summary
echo "-----------" >> $summary
echo Full training >> $summary
cat $full_summary >> $summary

echo "-----------"
echo "| SUMMARY |"
echo "-----------"
cat $summary