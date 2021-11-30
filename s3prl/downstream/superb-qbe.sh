#!/bin/bash

set -e

usage="The runfile for SUPERB Benchmark's Query by Example task

USAGE
    $0 -u UPSTREAM -q QUESST14 -p EXPS_ROOT [-h] [-d DIST1] [-d DIST2] ... [-l LAYER1] [-l LAYER2] ... [-o OVERRIDE]

    This runfile handles the distance function (DIST) search and the layer exploration (LAYER) for different upstreams when
    running Dynamic Time Warping (DTW). The runfile will run through all pairs of DIST and LAYER on the Dev set, and use the
    best pair to evaluate on the Test set. All the results, including all pairs on Dev and the best pair on Test, will be
    reported in a summary file. The runfile is stateful and fault-tolerance. If a run fails at the middle, you can simply
    restart the run with the exactly same command. If a pair was already evaluated, it will be skipped in the second run.

-u UPSTREAM (required)
    The entries defined in s3prl.hub. eg. wav2vec2, hubert, wav2vec2_large_ll60k... etc

-q QUESST14 (required)
    The root directory of Quesst14 dataset

-p EXPS_ROOT (required)
    All the experiment directories related to this benchmark will be located under UPSTREAM_DIR,
    which is either EXPS_ROOT/TASK/UPSTREAM or EXPS_ROOT/TASK/UPSTREAM/OVERRIDE (if -o is provided)
    You can find 'UPSTREAM_DIR/summary' for the exploration results.

-d DIST1 -d DIST2 (optional, str)
    Default: Only 'cosine_exp'
    The distance functions to explore, available options: cosine, cityblock, euclidean, cosine_exp, cosine_neg_log
    In our experience, cosine_exp is always the best for any upstream.

-l LAYER1 -l LAYER2 (optional, int)
    Default: all the layers of an upstream
    The layers of an upstream you want to explore. eg. -l 3 -l 4 for the 3-rd and 4-th layers.

-o OVERRIDE (optional)
    Default: empty
    Can be used to override any default fields in the args and the config file

-h
    Print this help message.
"

# Parse options
while getopts "u:q:p:o:d:l:h" flag
do
    case "${flag}" in
        u)
            upstream=${OPTARG}
            ;;
        q)
            quesst14=${OPTARG}
            ;;
        p)
            exps_root=${OPTARG}
            ;;
        o)
            override=${OPTARG}
            ;;
        d)
            dist_array+=("$OPTARG")
            ;;
        l)
            layer_array+=("$OPTARG")
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

if [ -z "$upstream" ] || [ ! -d "$quesst14" ] || [ -z "$exps_root" ]; then
    printf "$usage"
    exit 2
fi

if [ -z "$dist_array" ]; then
    dist_array="cosine_exp"
else
    dist_array="${dist_array[*]}"
fi

if [ -z "$layer_array" ]; then
    n_layers_file=$(mktemp)
    python3 utility/get_layer_num.py $upstream $n_layers_file
    n_layers=$(cat $n_layers_file)
    rm $n_layers_file
    layer_array="$(seq 0 $((n_layers-1)))"
else
    layer_array="${layer_array[*]}"
fi

start=$SECONDS
upstream_dir=$exps_root/QBE/$upstream
dev_dir=$upstream_dir/dev/; mkdir -p $dev_dir
dev_summary=$dev_dir/summary; [ -f $dev_summary ] && rm $dev_summary
echo "Using distance functions: $dist_array"
echo "Using $upstream layers: $layer_array"
echo "Start exploration with the Dev set, the summary will be at $dev_summary"

function get_eval_result() {
    if [ $# -ne 1 ]; then
        echo $FUNCNAME EXPDIR
        exit 2
    fi
    local expdir=$1

}

function single_trial() {
    if [ $# -ne 4 ]; then
        echo "$FUNCNAME EXPDIR LAYER DIST dev|test"
        exit 2
    fi
    local expdir=$1
    local layer=$2
    local dist=$3
    local mode=$4

    [ $mode == "dev" ] && local split="dev"
    [ $mode == "test" ] && local split="eval"

    cd $s3prl
    if [ ! -f $expdir/benchmark.stdlist.xml ]; then
        python3 run_downstream.py -m evaluate -t $mode -u $upstream -d quesst14_dtw -p $expdir \
            -l $layer -o config.downstream_expert.dtwrc.dist_method=$dist,,config.downstream_expert.datarc.dataset_root=$quesst14
    else
        echo "DTW results found. Skip DTW..."
    fi

    if [ ! -f $expdir/summary ] || [ -z "$(cat $expdir/summary)" ]; then
        cd $quesst14/scoring
        ./score-TWV-Cnxe.sh $s3prl/$expdir groundtruth_quesst14_$split -10
        cd $s3prl
        local score_file=$expdir/score.out
        if [ -f $score_file ]; then
            if [[ "$(cat $score_file)" =~ maxTWV:[[:space:]]+(.+)[[:space:]]+Threshold ]]; then
                echo "mtwv ${BASH_REMATCH[1]}" > $expdir/summary
            fi
        fi
    else
        echo "Scoring results found. Skip scoring..."
    fi

    echo "Results: $(cat $expdir/summary)"
}

s3prl=$(pwd)
for dist in $dist_array;
do
    for layer in $layer_array;
    do
        expdir=$upstream_dir/dev/layer_${layer}/$dist/; mkdir -p $expdir
        echo "[Dev] Trying $upstream with the layer $layer and the distance function $dist..."
        echo "Summary file will be at $expdir/summary"
        single_trial $expdir $layer $dist "dev"
    done
done

cd $s3prl
best_summary=$(grep mtwv $dev_dir/*/*/summary /dev/null | sort -grk 2 | head -n 1 | cut -d ":" -f 1)
if [[ "$best_summary" =~ $dev_dir/layer_(.+)/(.+)/summary ]]; then
    best_layer=${BASH_REMATCH[1]}
    best_dist=${BASH_REMATCH[2]}
fi
echo $best_layer > $upstream_dir/best_layer
echo $best_dist > $upstream_dir/best_dist

expdir=$upstream_dir/test/layer_${layer}/$dist/; mkdir -p $expdir
echo "[Test] Evaluate $upstream with the best layer $best_layer and the best distance function $best_dist"
echo "Summary file will be at $expdir/summary"
single_trial $expdir $layer $dist "test"

summary=$upstream_dir/summary;
[ -f $summary ]; rm $summary
echo "DEV (sorted by MTWV)" >> $summary
grep mtwv $upstream_dir/dev/*/*/summary /dev/null | sort -gk 2 >> $summary
echo "TEST (sorted by MTWV)" >> $summary
grep mtwv $upstream_dir/dev/*/*/summary /dev/null | sort -gk 2 >> $summary
echo "TIME" >> $summary
echo "$((SECONDS - start)) seconds" >> $summary
