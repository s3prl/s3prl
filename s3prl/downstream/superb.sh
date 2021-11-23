#!/bin/bash

set -e

supported_tasks="PR, KS, IC, SID, ASR, SF"
usage="The runfile for SUPERB Benchmark
    This runfile handles the learning rate (lr) search in the benchmark when training the downstream models. Since different upstreams
    (SSL models) need different suitable lr. Without the careful search, the final ranking can have huge differences. However, a single
    run for each upstream/downstream/lr pair takes a long time to fully converge, which can be unacceptable for most of the users. Hence,
    it is convenient and effective to determine the best lr with only the partial of training, and finish the full training only on the
    found best lr to get the best result. This can save lots of time and still get valid results. All the tasks except QBE (which does
    not involve training) can be driven by this runfile. The runfile currently only supports to benchmark with a single GPU.

USAGE
    $0 -u UPSTREAM -t TASK -p EXPS_ROOT [-h] [-o OVERRIDE] [-r EXPLORE_RATIO] [-l LR1] [-l LR2] ...

    The runfile is stateful and fault-tolerant. If an runfile execution was terminated, you can simply re-run with the exactly same command.
    The runfile will automatically determine where it was stopped and resume from there without any duplicated training. At the end of this
    execution, the runfile will report the summary of this run, including the dev results for all the lr search and the test result on the
    best lr. You can also re-check the summary message again with the exactly same command, which will skip all the training and directly
    report the summary. Usually, you only need to specify -u, -t and -p, since most of the configurations have proper default values, including
    the EXPLORE_RATIO and LR. Hence, -o, -r, -l are only needed when you want to explore more settings. eg. learning rates not included in the
    default setting, smaller EXPLORE_RATIO to further reduce exploration time.

UPSTREAM (required)
    The entries defined in s3prl.hub. eg. wav2vec2, hubert, wav2vec2_large_ll60k... etc

TASK (required)
    The task abbreviation: $supported_tasks

EXPS_ROOT (required)
    All the experiment directories related to this benchmark will be located under
    EXPS_ROOT/TASK/UPSTREAM or EXPS_ROOT/TASK/UPSTREAM/OVERRIDE (if -o is provided)

EXPLORE_RATIO (optional)
    Default: 0.05
    The percentage of the full training optimization steps for the learning rate search

LR1 LR2 ... (optional)
    Default: Each task has different default learning rates to explore
    If provided, will only search through these learning rates
    eg. -l 1e-3 -l 1e-4

OVERRIDE (optional)
    Default: empty
    Can be used to override any default fields in the args and the config file
    eg. args.upstream_layer_selection=3 to use the 3-rd layer as the representation to benchmark
        (Default use all layers and train the weighted-sum on them.)
"

# Parse options
while getopts "u:t:p:o:r:l:h" flag
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
explore_dir=$upstream_dir/optimize_ratio_$optimize_ratio; mkdir -p $explore_dir
echo "The results will be saved at $explore_dir"
for lr in $lrs;
do
    expdir=$explore_dir/lr_$lr; mkdir -p $expdir
    echo "Try learning $lr... The results will be saved at $expdir"
    single_trial "$expdir" "$upstream" "$(parse_override $optimize_ratio $lr $override)" false
done

explore_summary=$explore_dir/summary; [ -f "$explore_summary" ] && rm $explore_summary
echo "Report exploration result at $explore_summary"
for expdir in $(ls -d $explore_dir/*);
do
    eval_result=$(get_eval_result $expdir "dev")
    if [ ! -z "$eval_result" ]; then
        echo $(basename $expdir): $eval_result >> $explore_summary
    fi
done
cat $explore_summary

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

full_dir=$upstream_dir/optimize_ratio_1; mkdir -p $full_dir
expdir=$full_dir/lr_$best_lr; mkdir -p $expdir
echo "Final full training with learning rate $best_lr"
echo "The results will be saved at $expdir"
single_trial "$expdir" "$upstream" "$(parse_override 1 $best_lr $override)" true

echo "Report full training result..."
full_summary=$full_dir/summary; [ -f "$full_summary" ] && rm $full_summary
for expdir in $(ls -d $full_dir/*);
do
    eval_result=$(get_eval_result $expdir "test")
    if [ ! -z "$eval_result" ]; then
        echo $(basename $expdir): $eval_result >> $full_summary
    fi
done

summary=$upstream_dir/summary
echo "SUMMARY
TIME: $((SECONDS - start)) seconds

PARTIAL TRAINING (LEARNING RATE EXPLORATION)
$(cat $explore_summary)

BEST LEARNING RATE
$(cat $explore_dir/best_lr)

FULL TRAINING
"$(cat $full_summary)"
" > $summary

echo
cat $summary
