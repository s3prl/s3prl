#!/bin/bash
#   *********************************************************************************************"""
#   FileName     [ score.sh ]
#   Synopsis     [ Speaker Diarization Scoring, use NIST scoring metric ]
#   Source       [ Refactored From https://github.com/hitachi-speech/EEND ]
#   Author       [ Jiatong Shi ]
#   Copyright    [ Copyright(c), Johns Hopkins University ]
#   *********************************************************************************************"""

set -e
set -x

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <expdir> <test_set>"
  echo "e.g., ./downstream/diarization/score.sh result/downstream/test data/test"
  exit 1
fi

scoring_dir="$1/scoring"
infer_dir="${scoring_dir}/predictions"
test_set="$2"
# directory where you cloned dscore (https://github.com/ftshijt/dscore)
dscore_dir=/groups/leo1994122701/dscore

frame_shift_file=$1/frame_shift
if [ -f $frame_shift_file ]; then
    frame_shift=$(cat $frame_shift_file)
else
    echo "[Warning] File not found: $frame_shift_file. Degenerate to use frame shift 160. "`
         `"If you suspect the downstream model was not trained by the labels in frame shift 160, please "`
         `"create a file $frame_shift_file with a single number: the trained model's frame shift. "`
         `"Before the PR https://github.com/s3prl/s3prl/pull/202, the diarization models are always trained "`
         `"by frame shift 160. After this PR, models are trained by the label whose frame shift is upstream "`
         `"representation's downsample rate. See the PR page for more information."
    frame_shift=160
fi
sr=16000

echo "scoring at $scoring_dir"
scoring_log_dir=$scoring_dir/log
mkdir -p $scoring_log_dir || exit 1;
find $infer_dir -iname "*.h5" > $scoring_log_dir/file_list
for med in 1 11; do
    for th in 0.3 0.4 0.5 0.6 0.7; do
        python downstream/diarization/make_rttm.py --median=$med --threshold=$th \
            --frame_shift=${frame_shift} --subsampling=1 --sampling_rate=${sr} \
            $scoring_log_dir/file_list $scoring_dir/hyp_${th}_$med.rttm
        python ${dscore_dir}/score.py -r ${test_set}/rttm -s $scoring_dir/hyp_${th}_$med.rttm \
            > $scoring_dir/result_th${th}_med${med} 2>/dev/null || exit

        # NIST scoring
        # md-eval.pl \
        #     -r ${test_set}/rttm \
        #     -s $scoring_dir/hyp_${th}_$med.rttm > $scoring_dir/result_th${th}_med${med}_collar0 2>/dev/null || exit
    done
done

grep OVER $scoring_dir/result_th0.[^_]*_med[^_]* \
     | sort -nrk 4
