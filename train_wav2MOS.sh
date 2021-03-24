# name of the experiment
EXP_NAME=wav2MOS_segment_cpc_testing
# which upstream to run. Ex. wav2vec2, npc, TERA, ...
UPSTREAM=cpc
# which downstream to run.
DOWNSTREAM=mos_prediction_segment

python3 run_downstream.py -m train -n $EXP_NAME -u $UPSTREAM -d $DOWNSTREAM