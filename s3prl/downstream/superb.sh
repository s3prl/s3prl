#!/bin/bash

set -e

supported_tasks="PR, KS, IC, SID, ER, ASR, SF"
usage="The runfile for SUPERB Benchmark

USAGE
    $0 -u UPSTREAM -t TASK -p EXPS_ROOT [-h] [-n] [-o OVERRIDE] [-r EXPLORE_RATIO] [-l LR1] [-l LR2] ...

    This runfile handles the learning rate (lr) search in the benchmark when training the downstream models. Since different upstreams
    (SSL models) need different suitable lr. Without the careful search, the final ranking can have huge differences. However, a single
    run for each upstream/downstream/lr pair takes a long time to fully converge, which can be unacceptable for most of the users. Hence,
    it is convenient and effective to determine the best lr with only the partial of training, and finish the full training only on the
    found best lr to get the best result. This can save lots of time and still get valid results. The runfile handles two stages:

    Stage 1. Learning rate exploration with only partial of the original training.
    Stage 2. Train the model to converge with the best learning rate found in Stage 1.

    All the tasks in SUPERB except QBE (which does not involve training) can be driven by this runfile. The runfile currently only supports
    to benchmark with a single GPU. The runfile is stateful and fault-tolerant. If an runfile execution was terminated, you can simply re-run
    with the exactly same command. The runfile will automatically determine where it was stopped and resume from there without any duplicated
    training. At the end of this execution, the runfile will report the summary of this run, including the dev results for all the lr search
    during the partial training, and the test result on the best lr after the full tarining. You can also re-check the summary message again
    with the exactly same command, which will skip all the training and directly report the summary.

    Usually, you only need to specify -u, -t, -p, since most of the configurations have proper default values, including EXPLORE_RATIO and LR.
    Hence, -o, -r, -l are only needed when you want to explore more settings.
    Eg.
        1. Set -r to smaller EXPLORE_RATIO to further reduce the exploration time.
        2. Set -l to try learning rates not included in the default setting.
        3. Set -o to override any default fields in the command-line argument or the config file.

    Here is an example command to benchmark HuBERT Base model on Keyword Spotting.

    ./downstream/superb.sh -u hubert -t KS -p result/superb/

    WARNING: Before you run the above command, please first make sure you already followed each task's 'Prepare Data' subsection in the
    SUPERB documentation: https://github.com/s3prl/s3prl/blob/master/s3prl/downstream/docs/superb.md

    By default, weighted-sum on all the hidden states are trained along with the downstream model. You can benchmark an upstream's layer-wise
    performance via -o (which can overrides any command-line argument and config file field of run_downstream.py) to select a specific layer.
    The following command benchmarks the 3-rd layer of HuBERT Base on Keyword Spotting.

    ./downstream/superb.sh -u hubert -t KS -p result/superb/ -o args.upstream_layer_selection=3

-u UPSTREAM (required)
    The entries defined in s3prl.hub. eg. wav2vec2, hubert, wav2vec2_large_ll60k... etc

-t TASK (required)
    The task abbreviation: $supported_tasks

-p EXPS_ROOT (required)
    All the experiment directories related to this benchmark will be located under UPSTREAM_DIR,
    which is either EXPS_ROOT/TASK/UPSTREAM or EXPS_ROOT/TASK/UPSTREAM/OVERRIDE (if -o is provided)
    Typicall you will want to see 'UPSTREAM_DIR/summary' for the lr exploration and full training
    results.

-o OVERRIDE (optional)
    Default: empty
    Can be used to override any default fields in the args and the config file
    eg. args.upstream_layer_selection=3 to use the 3-rd layer as the representation to benchmark
        (Default use all layers and train the weighted-sum on them.)

-r EXPLORE_RATIO (optional)
    Default: 0.05
    The percentage of the full training optimization steps for the learning rate search

-l LR1 -l LR2 ... (optional)
    Default: Each task has different default learning rates to explore
    If provided, will only search through these learning rates
    eg. -l 1e-3 -l 1e-4

-n (advanced)
    Stop right after the learning rate exploration (Stage 1) and before the full training (Stage 2).
    Sometimes you are not sure what is the suitable learning rate exploration range, and wish to
    manually check the exploration results before continue to the full training (Stage 2) which costs
    really long time (and perhaps lots of money). You can stop right before Stage 2 with -n option.
    After manual check, you might want to explore more learning rates, or go on the full training
    if everything is as expected. Here is an example.

    Basic command: should be the same across the following runs
    ./downstream/superb.sh -u hubert -t KS -p result/superb/

    First explore:
    ./downstream/superb.sh -u hubert -t KS -p result/superb/ -n -l 1e-2 -l 1e-3 -n

    Second explore:
    ./downstream/superb.sh -u hubert -t KS -p result/superb/ -n -l 1e-4 -l 1e-5 -n

    After manually check the exploration results, continue the full training:
    ./downstream/superb.sh -u hubert -t KS -p result/superb/ -n -l 1e-2 -l 1e-3 -l 1e-4 -l 1e-5

    Note that in the final command, you do not need to specify -n, and all the required -l can be found
    in 'UPSTREAM_DIR/summary', which summarizes all the explored learning rates and their results.

-h
    Print this help message.

"

# Parse options
while getopts "u:t:p:o:r:l:nh" flag
do
    case "${flag}" in
        u)
            upstream=${OPTARG}
            ;;
        t) 
            task=${OPTARG}
            ;;
        p)
            exps_root=${OPTARG}
            ;;
        o)
            override=${OPTARG}
            ;;
        r)
            explore_ratio=${OPTARG}
            ;;
        l)
            lr_array+=("$OPTARG")
            ;;
        n)
            no_stage2=true
            ;;
        h)
            printf "$usage"
            exit 2
            ;;
        ?)
            printf "$usage"
            exit 2
            ;;
    esac
done

if [ -z "$upstream" ]|| [ -z "$task" ] || [ -z "$exps_root" ]; then
    printf "$usage"
    exit 2
fi

# Set task specific configuration
case "$task" in
    PR)
        config_bash=downstream/ctc/pr.sh
        ;;
    KS)
        config_bash=downstream/speech_commands/ks.sh
        ;;
    IC)
        config_bash=downstream/fluent_commands/ic.sh
        ;;
    SID)
        config_bash=downstream/voxceleb1/sid.sh
        ;;
    ER)
        config_bash=downstream/emotion/er.sh
        ;;
    ASR)
        config_bash=downstream/asr/asr.sh
        ;;
    SF)
        config_bash=downstream/ctc/sf.sh
        ;;
    *)
        echo "Invalid task "$task""
        echo "Supported -t arguments: ${supported_tasks}"
        exit 2
        ;;
esac
source $config_bash

if [[ $(type -t single_trial) != function ]] ||
   [[ $(type -t get_eval_result) != function ]] ||
   [ -z "$default_lrs" ] ||
   [ -z "$metric_higher_better" ];
then
    echo "Task-specific configuration bash is not well configured"
    echo "Please check $config_bash"
    exit 1
fi

# Set default values
if [ -z "$explore_ratio" ]; then
    if [ -z "$default_explore_ratio" ]; then
        explore_ratio=0.05
    else
        explore_ratio="$default_explore_ratio"
    fi
fi
if [ -z "$lr_array" ]; then
    lrs="$default_lrs"
else
    lrs="${lr_array[*]}"
fi

# End parsing, start benchmarking
start=$SECONDS

upstream_dir=$exps_root/$task/$upstream
if [ ! -z "$override" ]; then
    upstream_dir=$upstream_dir/$override
fi
mkdir -p $upstream_dir
summary=$upstream_dir/summary
[ -f $summary ] && rm $summary
echo "SUMMARY" >> $summary
echo "You can see the benchmark summary at $summary"

function parse_override() {
    if [ "$#" -lt "2" ]; then
        echo "Usage: parse_override OPTIMIZE_RATIO LR [OVERRIDE]"
        exit 2
    fi

    local optimize_ratio=$1
    local lr=$2
    local override=$3

    if [ ! -z $override ]; then
        override="$override,,"
    fi
    echo "${override}config.runner.optimize_ratio=${optimize_ratio},,config.optimizer.lr=${lr}"
}

# Explore learning rate
optimize_ratio=$explore_ratio
echo "Exploring learning rate $lrs with optimization ratio $optimize_ratio"
explore_dir=$upstream_dir/partial_training; mkdir -p $explore_dir
echo "The results will be saved at $explore_dir"
for lr in $lrs;
do
    expdir=$explore_dir/lr_$lr; mkdir -p $expdir
    echo "Try learning $lr... The results will be saved at $expdir"
    single_trial "$expdir" "$upstream" "$(parse_override $optimize_ratio $lr $override)" false
done

explore_summary=$explore_dir/summary; [ -f "$explore_summary" ] && rm $explore_summary
echo "Report exploration result to $explore_summary"
for expdir in $(ls -d $explore_dir/*/);
do
    eval_result=$(get_eval_result $expdir "dev")
    if [ ! -z "$eval_result" ]; then
        echo $(basename $expdir): $eval_result >> $explore_summary
    fi
done
echo "
PARTIAL TRAINING (LEARNING RATE EXPLORATION)
$(cat $explore_summary)" >> $summary
cat $summary

if [ ! -z $no_stage2 ]; then
    exit 0
fi

echo "Picking the best learning rate..."
if [ $metric_higher_better = true ]; then
    reverse_sort="-r"
else
    reverse_sort=""
fi
best_dev=$(cat $explore_summary | sort $reverse_sort -gk 3 | head -n 1)
[[ "$best_dev" =~ lr_(.*): ]] && best_lr=${BASH_REMATCH[1]}
if [ -z "$best_lr" ]; then
    echo "The training/dev/test during the learning rate exploration were not completed or files are corrupted"
    echo "Please delete $upstream_dir and try again"
    exit 1
fi
echo "Best learning rate: $best_lr"
echo "Save to $explore_dir/best_lr"
echo $best_lr > $explore_dir/best_lr

full_dir=$upstream_dir/full_training; mkdir -p $full_dir
expdir=$full_dir/lr_$best_lr; mkdir -p $expdir
echo "Final full training with learning rate $best_lr"
echo "The results will be saved at $expdir"
single_trial "$expdir" "$upstream" "$(parse_override 1 $best_lr $override)" true

full_summary=$full_dir/summary; [ -f "$full_summary" ] && rm $full_summary
echo "Report full training result to $full_summary"
for expdir in $(ls -d $full_dir/*/);
do
    eval_result=$(get_eval_result $expdir "test")
    if [ ! -z "$eval_result" ]; then
        echo $(basename $expdir): $eval_result >> $full_summary
    fi
done

echo "Report the entire benchmark summary at $summary"
echo "
BEST LEARNING RATE
$(cat $explore_dir/best_lr)

FULL TRAINING
$(cat $full_summary)

TIME
$((SECONDS - start)) seconds" >> $summary

echo
cat $summary
