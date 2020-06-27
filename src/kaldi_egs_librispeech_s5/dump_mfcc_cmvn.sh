#!/bin/bash

data_root=/media/andi611/1TBSSD/kaldi/egs/librispeech/s5 ## You'll want to change this path to something that will work on your system.

mkdir $data_root/mfcc_cmvn/

for part in dev_clean test_clean train_clean_100 train_clean_360 train_other_500; do
  mkdir $data_root/mfcc_cmvn/$part/
  apply-cmvn --utt2spk=ark:$data_root/data/$part/utt2spk scp:$data_root/mfcc/cmvn_$part.scp scp:$data_root/data/$part/feats.scp ark,scp:$data_root/mfcc_cmvn/$part/mfcc_cmvn_nd.ark,$data_root/mfcc_cmvn/$part/mfcc_cmvn_nd.scp | add-deltas --delta-order=2 ark:$data_root/mfcc_cmvn/$part/mfcc_cmvn_nd.ark ark:$data_root/mfcc_cmvn/$part/mfcc_cmvn.ark
done

du -sh $data_root/mfcc_cmvn/*
echo "Done!"