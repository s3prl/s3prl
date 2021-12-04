#!/bin/bash

set -e

usage="The runfile for SUPERB Benchmark's Query by Example task

USAGE
    $0 -u UPSTREAM -q QUESST14 -p EXPS_ROOT [-h] [-s STAGE1] [-s STAGE2] [-d DIST1] [-d DIST2] ... [-l LAYER1] [-l LAYER2] ... [-o OVERRIDE]

    This runfile handles the distance function (DIST) search and the layer exploration (LAYER) for different upstreams when
    running Dynamic Time Warping (DTW). The runfile will run through all pairs of DIST and LAYER on the Dev set, and use the
    best pair to evaluate on the Test set. All the results, including all pairs on Dev and the best pair on Test, will be
    reported in a summary file. The runfile is stateful and fault-tolerance. If a run fails at the middle, you can simply
    restart the run with the exactly same command. If a pair was already evaluated, it will be skipped in the second run.

    The runfile has two stages. Stage 1 extract features from the upstream model and store the features. Stage 2 load these
    features back and run DTW on them. Usually, you will want to run Stage 1 with GPU, and run Stage 2 with only CPU. Stage 1
    can typically finish in 10 minutes, while Stage 2 can take several hours. However, Stage 2 cannot be optimized by GPU hence
    giving GPU resource to it is a complete waste of your computing budget.

-u UPSTREAM (required)
    The entries defined in s3prl.hub. eg. wav2vec2, hubert, wav2vec2_large_ll60k... etc

-q QUESST14 (required)
    The root directory of Quesst14 dataset

-p EXPS_ROOT (required)
    All the experiment directories related to this benchmark will be located under UPSTREAM_DIR,
    which is either EXPS_ROOT/TASK/UPSTREAM or EXPS_ROOT/TASK/UPSTREAM/OVERRIDE (if -o is provided)
    You can find 'UPSTREAM_DIR/summary' for the exploration results.

-s STAGE (optional, int)
    Default: 1 and 2
    Available options: 1 or 2

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
while getopts "u:q:s:p:d:l:o:h" flag
do
    case "${flag}" in
        u)
            upstream=${OPTARG}
            ;;
        q)
            quesst14=${OPTARG}
            ;;
        s)
            [ "${OPTARG}" == 1 ] && stage1=true;
            [ "${OPTARG}" == 2 ] && stage2=true;
            ;;
        p)
            exps_root=${OPTARG}
            ;;
        d)
            dist_array+=("$OPTARG")
            ;;
        l)
            layer_array+=("$OPTARG")
            ;;
        o)
            override=${OPTARG}
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

# set default values
if [ -z "$dist_array" ]; then
    dist_array="cosine_exp"
else
    dist_array="${dist_array[*]}"
fi
if [ -z "$stage1" ] && [ -z "$stage2" ]; then
    stage1=true;
    stage2=true;
else
    [ -z "$stage1" ] && stage1=false
    [ -z "$stage2" ] && stage2=false
fi

s3prl=$(pwd)
upstream_dir=$exps_root/QBE/$upstream
dev_dir=$upstream_dir/dev; mkdir -p $dev_dir
test_dir=$upstream_dir/test; mkdir -p $test_dir
start=$SECONDS

summary=$upstream_dir/summary
[ -f $summary ] && echo "The experiment seems to complete since $summary exists." && cat $summary && exit 1

if [ $stage1 == true ]; then
    if [ -z "$layer_array" ]; then
        n_layers_file=$(mktemp)
        python3 utility/get_layer_num.py $upstream $n_layers_file
        n_layers=$(cat $n_layers_file)
        rm $n_layers_file
        layer_array="$(seq 0 $((n_layers-1)))"
    else
        layer_array="${layer_array[*]}"
    fi
    echo "$layer_array" > $upstream_dir/layers

    echo "Using $upstream layers: $layer_array"
    for layer in $layer_array;
    do
        function extract_features() {
            [ $# -ne 1 ] && echo $FUNCNAME dev|test && exit 2
            local mode=$1
            expdir=$(eval echo \${${mode}_dir})/layer_$layer

            echo "Extracting ${upstream}'s layer $layer features to ${expdir} ..."
            if [ "$(ls $expdir/*.pkl | wc -l)" -lt 2 ]; then
                python3 run_downstream.py -m evaluate -t "$mode" -u $upstream -l $layer -d quesst14_dtw -p $expdir \
                    -o config.downstream_expert.datarc.dataset_root=$quesst14,,config.downstream_expert.two_stages=True
                [ "$mode" == "test" ] && rm $expdir/*doc*.pkl
            else
                echo "Extracted features found. Skip Stage 1."
            fi
            return 0
        }
        extract_features "dev"
        extract_features "test"
    done
fi

if [ $stage2 == true ]; then
    [ -f $upstream_dir/layers ] && layer_array=$(cat $upstream_dir/layers)
    [ -z "$layer_array" ] && echo "Please run Stage 1 first: -s 1" && exit 2
    dev_summary=$dev_dir/summary; [ -f $dev_summary ] && rm $dev_summary
    test_summary=$test_dir/summary; [ -f $test_summary ] && rm $test_summary
    echo "Using $upstream layers: $layer_array"

    function single_trial() {
        if [ $# -ne 5 ]; then
            echo "$FUNCNAME QUERIES_DIR DOCS_DIR OUT_DIR DIST dev|eval"
            exit 2
        fi
        local queries_dir=$1
        local docs_dir=$2
        local out_dir=$3
        local dist=$4
        local split=$5

        if [ "$(ls $queries_dir/*quer*.pkl | wc -l)" -lt 2 ] || [ "$(ls $docs_dir/*doc*.pkl | wc -l)" -lt 2 ]; then
            echo "Please run Stage 1 first: -s 1"
            exit 2
        fi

        if [ ! -f $out_dir/benchmark.stdlist.xml ]; then
            python3 downstream/quesst14_dtw/dtw_utils.py $queries_dir $out_dir \
                --docs_dir $docs_dir -o config.downstream_expert.dtwrc.dist_method=$dist
        else
            echo "DTW results found. Skip DTW..."
        fi

        if [ ! -f $out_dir/summary ] || [ -z "$(cat $out_dir/summary)" ]; then
            cd $quesst14/scoring
            ./score-TWV-Cnxe.sh $s3prl/$out_dir groundtruth_quesst14_$split -10
            cd $s3prl
            local score_file=$out_dir/score.out
            if [ -f $score_file ]; then
                if [[ "$(cat $score_file)" =~ maxTWV:[[:space:]]+(.+)[[:space:]]+Threshold ]]; then
                    echo "mtwv ${BASH_REMATCH[1]}" > $out_dir/summary
                fi
            fi
        else
            echo "Scoring results found. Skip scoring..."
        fi

        echo "Results: $(cat $out_dir/summary)"
        return 0
    }

    echo "Using distance functions: $dist_array"
    echo "Start exploration with the Dev set, the summary will be at $dev_summary"

    for layer in $layer_array;
    do
        feat_dir=$dev_dir/layer_$layer
        for dist in $dist_array;
        do
            out_dir=$feat_dir/$dist; mkdir -p $out_dir;
            echo "[Dev] Trying $upstream with the layer $layer and the distance function $dist..."
            echo "Summary file will be at $out_dir/summary"
            single_trial $feat_dir $feat_dir $out_dir $dist "dev"
        done
        grep mtwv $dev_dir/*/*/summary /dev/null | sort -grk 2 > $dev_summary
    done

    echo "Finish Dev:"
    cat $dev_summary

    best_summary=$(cat $dev_summary | head -n 1 | cut -d ":" -f 1)
    if [[ "$best_summary" =~ $dev_dir/layer_(.+)/(.+)/summary ]]; then
        best_layer=${BASH_REMATCH[1]}
        best_dist=${BASH_REMATCH[2]}
    fi
    echo $best_layer > $upstream_dir/best_layer
    echo $best_dist > $upstream_dir/best_dist

    queries_dir=$test_dir/layer_${best_layer}
    docs_dir=$dev_dir/layer_${best_layer}
    out_dir=$queries_dir/$best_dist
    echo "[Test] Evaluate $upstream with the best layer $best_layer and the best distance function $best_dist"
    echo "Summary file will be at $out_dir/summary"
    single_trial $queries_dir $docs_dir $out_dir $dist "eval"
    grep mtwv $test_dir/*/*/summary /dev/null | sort -gk 2 >> $test_summary

    echo "SUMMARY" >> $summary
    echo "DEV (sorted by MTWV)" >> $summary
    cat $dev_summary >> $summary
    echo "TEST (sorted by MTWV)" >> $summary
    cat $test_summary >> $summary
    echo "TIME" >> $summary
    echo "$((SECONDS - start)) seconds" >> $summary

    echo "QBE completes. Remove dumped features to save space."
    rm $dev_dir/*/*.pkl
    rm $test_dir/*/*.pkl
fi
