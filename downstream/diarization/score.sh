"""*********************************************************************************************"""
#   FileName     [ score.sh ]
#   Synopsis     [ Speaker Diarization Scoring, use NIST scoring metric ]
#   Source       [ Refactored From https://github.com/hitachi-speech/EEND ]
#   Author       [ Jiatong Shi ]
#   Copyright    [ Copyleft(c), Johns Hopkins University ]
"""*********************************************************************************************"""

scoring_dir="downstream/diarization/scoring"
infer_dir="downstream/diarization/scoring/predictions"
test_set="downstream/diarization/data/test"

frame_shift=160
sr=16000

if [ $stage -le 4 ]; then
    echo "scoring at $scoring_dir"
	scoring_log_dir=$scoring_dir/log
    mkdir -p $scoring_log_dir || exit 1;
	find $infer_dir -iname "*.h5" > $scoring_log_dir/file_list
    for med in 1 11; do
        for th in 0.3 0.4 0.5 0.6 0.7; do
            make_rttm.py --median=$med --threshold=$th \
                --frame_shift=${frame_shift} --subsampling=1 --sampling_rate=${sr} \
                $scoring_log_dir/file_list $scoring_dir/hyp_${th}_$med.rttm
            md-eval.pl -c 0.25 \
                -r ${test_set}/rttm \
                -s $scoring_dir/hyp_${th}_$med.rttm > $scoring_dir/result_th${th}_med${med}_collar0.25 2>/dev/null || exit
        done
    done
fi

grep OVER $scoring_dir/result_th0.[^_]*_med[^_]*_collar0.25 | grep -v nooverlap | sort -nrk 7 | tail -n 1 