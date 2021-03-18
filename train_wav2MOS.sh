# name of the experiment
EXP_NAME=wav2MOS_linear_wav2vec2
# which upstream to run. Ex. wav2vec2, npc, TERA, ...
UPSTREAM=wav2vec2
# which downstream to run.
DOWNSTREAM=wav2MOS_linear

python3 run_downstream.py -m train -n $EXP_NAME -u $UPSTREAM -d $DOWNSTREAM -f