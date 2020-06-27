#!/bin/bash

data_root=/media/andi611/1TBSSD/kaldi/egs/librispeech/s5 ## You'll want to change this path to something that will work on your system.

mkdir $data_root/fmllr_cmvn/

for part in dev_clean test_clean train_clean_100 train_clean_360 train_other_500; do
  mkdir $data_root/fmllr_cmvn/$part/
  apply-cmvn --utt2spk=ark:$data_root/fmllr/$part/utt2spk ark:$data_root/fmllr/$part/data/cmvn_speaker.ark scp:$data_root/fmllr/$part/feats.scp ark:$data_root/fmllr_cmvn/$part/fmllr_cmvn.ark
done

du -sh $data_root/fmllr_cmvn/*
echo "Done!"