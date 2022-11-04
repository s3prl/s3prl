#!/bin/bash

set -e

supported_tasks="PR, KS, IC, SID, ER, ASR, SF, ASV, SD, VC, SE"
usage="The runfile for SUPERB Benchmark

USAGE
    $0 -u UPSTREAM -t TASK -p EXPS_ROOT [-h] [-o OVERRIDE] [-r EXPLORE_RATIO] [-l LR1] [-l LR2] ... [-s STAGE1] [-s STAGE2] ... [-e EARLY_STOP]

    This runfile handles the learning rate (lr) search in the benchmark when training the downstream models. Since different upstreams
    (SSL models) need different suitable lr. Without the careful search, the final ranking can have huge differences. However, a single
    run for each upstream/downstream/lr pair takes a long time to fully converge, which can be unacceptable for most of the users. Hence,
    it is convenient and effective to determine the best lr with only the partial of training, and finish the full training only on the
    found best lr to get the best result. This can save lots of time and still get valid results. The runfile handles two stages:

    Stage 1. Learning rate exploration with only partial of the original training.
    Stage 2. Train the model to converge with the best learning rate found in Stage 1.

    All the tasks in SUPERB except QBE (please use downstream/superb-qbe.sh) can be driven by this runfile. The runfile currently only supports
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
    which is either EXPS_ROOT/TASK/UPSTREAM or EXPS_ROOT/TASK/UPSTREAM__OVERRIDE (if -o is provided)
    Typicall you will want to see 'UPSTREAM_DIR/summary' for the lr exploration and full training
    results.

-o OVERRIDE (optional)
    Default: empty
    Can be used to override any default fields in the args and the config file
    eg. args.upstream_layer_selection=3 to use the 3-rd layer as the representation to benchmark
        (Default use all layers and train the weighted-sum on them.)

-r EXPLORE_RATIO (float, optional)
    Default: task-dependent
    The percentage of the full training optimization steps for the learning rate search

-l LR1 -l LR2 ... (float, optional)
    Default: task-dependent
    If provided, will only search through these learning rates
    eg. -l 1e-3 -l 1e-4

-s STAGE1 -s STAGE2 (int, optional)
    Default: both stage 1 and stage 2 are executed, equivalent to -s 1 -s 2, except ASV only runs -s 2
    Sometimes you are pretty sure about the best learning rate and wish to skip Stage 1 (learning
    rate exploration); while sometimes you are completely not sure about the suitable lr exploration
    range and wish to proceed Stage 2 (full training) after manually checking the Stage 1 results.
    The -s option gives you the flexibility on the stages you wish to run. If no -s is assigned (default),
    both stage 1 and stage 2 are executed. If any -s is given, only the assigned stage will be executed.

-h
    Print this help message.

"

# Parse options
while getopts "u:t:p:o:r:l:s:e:h" flag
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
        s)
            [ "$OPTARG" == "1" ] && stage1=true
            [ "$OPTARG" == "2" ] && stage2=true
            ;;
        e)
            early_stop_steps=${OPTARG}
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
    SD)
        config_bash=downstream/diarization/sd.sh
        ;;
    ASV)
        config_bash=downstream/sv_voxceleb1/asv.sh
        ;;
    VC)
        config_bash=downstream/a2o-vc-vcc2020/vc.sh
        ;;
    SE)
        config_bash=downstream/enhancement_stft/se.sh
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
if [ -z "$stage1" ] && [ -z "$stage2" ]; then
    stage1=true
    stage2=true
fi

# End parsing, start benchmarking
start=$SECONDS

upstream_dir=$exps_root/$task/$upstream
if [ ! -z "$override" ]; then
    upstream_dir=${upstream_dir}/${override//,,/__}
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

if [ "$stage1" = true ]; then
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
    echo "Report exploration result to $explore_summary"
    for lr in $lrs;
    do
        expdir=$explore_dir/lr_$lr/
        eval_result=$(get_eval_result $expdir "dev")
        if [ ! -z "$eval_result" ]; then
            echo $(basename $expdir): $eval_result >> $explore_summary
        fi
    done
    printf "PARTIAL TRAINING (LEARNING RATE EXPLORATION)\n"`
        `"$(cat $explore_summary)\n" >> $summary

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
    printf "BEST LEARNING RATE\n"`
        `"$(cat $explore_dir/best_lr)\n"`
        `"TIME\n"`
        `"$((SECONDS - start)) seconds\n" >> $summary

    echo
    echo "Finish Stage 1. Report the summary at $summary:"
    echo
    cat $summary
    echo
fi

if [ "$stage2" = true ]; then
    full_dir=$upstream_dir/optimize_ratio_1; mkdir -p $full_dir
    if [ "$stage1" = true ]; then
        [ -z "${best_lr}" ] && echo "Files corrupted. Please delete ${upstream_dir} and try again." && exit 1
        echo "Stage 1 was executed. Run Stage 2 with the found best learning rate: ${best_lr}"
        full_lrs="$best_lr"
    else
        echo "Stage 1 was not executed. Run Stage 2 with all the learning rate given by the -l option: ${lrs}"
        full_lrs="$lrs"
    fi

    for lr in $full_lrs;
    do
        lr_full_expdir=$full_dir/lr_$lr; mkdir -p $lr_full_expdir
        echo "Final full training with learning rate $lr"
        echo "The results will be saved at $lr_full_expdir"

        function copy_if_exists() {
            if [ $# -lt 3 ]; then
                echo "$FUNCNAME SRC_DIR TGT_DIR PATTERN [N_LATEST]"
                exit 2
            fi
            src_dir=$1
            tgt_dir=$2
            pattern=$3
            n_latest=$4
            echo "Copying files with the pattern $pattern from:"
            echo "$src_dir/ -> $tgt_dir/"
            if [ -d $src_dir ]; then
                if ls $src_dir | grep -q -E "$pattern"; then
                    files="$(ls -1t $src_dir | grep -E "$pattern")"
                    echo "Found files: $files"
                    if [ ! -z "$n_latest" ]; then
                        files=$(echo $files | head -n $n_latest)
                        echo "Only pick the $n_latest latest files: $files"
                    fi
                    for file in $files;
                    do
                        src_file=$src_dir/$file
                        echo "Copying $file"
                        cp -r $src_file $tgt_dir/
                    done
                else
                    echo "Files not found for the pattern "$pattern" in $dir"
                fi
            else
                echo "$dir does not exist"
            fi
        }

        if [ ! -z "$explore_dir" ]; then
            lr_explore_expdir=$explore_dir/lr_$lr
            copy_if_exists $lr_explore_expdir $lr_full_expdir "states-[0-9]+\.ckpt" 1
            copy_if_exists $lr_explore_expdir $lr_full_expdir "dev.*\.ckpt"
            copy_if_exists $lr_explore_expdir $lr_full_expdir "events\.out\.tfevents"
        fi

        override="$(parse_override 1 $lr $override)"
        if [ ! -z "$early_stop_steps" ]; then
            override="$override,,config.runner.total_steps=$early_stop_steps"
        fi
        single_trial "$lr_full_expdir" "$upstream" "$override" true
    done

    full_summary=$full_dir/summary; [ -f "$full_summary" ] && rm $full_summary
    echo "Report full training result to $full_summary"
    for expdir in $(ls -d $full_dir/*/);
    do
        eval_result=$(get_eval_result $expdir "test")
        if [ ! -z "$eval_result" ]; then
            echo $(basename $expdir): $eval_result >> $full_summary
        fi
    done

    printf "FULL TRAINING\n"`
        `"$(cat $full_summary)\n"`
        `"TIME\n"`
        `"$((SECONDS - start)) seconds\n" >> $summary

    echo
    echo "Finish Stage 2. Report the summary at $summary"
    echo
    cat $summary
    echo
fi
