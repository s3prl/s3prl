#!/bin/bash

data_root=/media/andi611/1TBSSD/kaldi/egs/librispeech/s5 ## You'll want to change this path to something that will work on your system.

mkdir $data_root/fbank_cmvn/

for part in dev_clean test_clean train_clean_100 train_clean_360 train_other_500; do
  mkdir $data_root/fbank_cmvn/$part/
  apply-cmvn --utt2spk=ark:$data_root/data/$part/utt2spk  ark:$data_root/fbank/cmvn_$part.ark scp:$data_root/data/$part/feats.scp ark:- | add-deltas --delta-order=0 ark:- ark:$data_root/fbank_cmvn/$part/fbank_cmvn.ark
done

du -sh $data_root/fbank_cmvn/*
echo "Done!"