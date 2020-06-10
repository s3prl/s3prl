#!/bin/bash

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
. ./path.sh ## Source the tools/utils (import the queue.pl)

gmmdir=exp/tri4b

for chunk in dev_clean test_clean train_clean_100 train_clean_360 train_other_500 ; do
    dir=fmllr/$chunk
    steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
        --transform-dir $gmmdir/decode_tgsmall_$chunk \
            $dir data/$chunk $gmmdir $dir/log $dir/data || exit 1

    compute-cmvn-stats --spk2utt=ark:data/$chunk/spk2utt scp:fmllr/$chunk/feats.scp ark:$dir/data/cmvn_speaker.ark
done