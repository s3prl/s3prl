#!/bin/bash
data_dir=/media/andi611/1TBSSD/kaldi/egs/librispeech/s5/data
dec_dir=/media/andi611/1TBSSD/pytorch-kaldi/exp/libri-liGRU-BASE-960-RA-1000/decode_test_clean_out_dnn2 # change this path to your exp dir
out_dir=/media/andi611/1TBSSD/pytorch-kaldi/exp/libri-liGRU-BASE-960-RA-1000 # change this path to your exp dir

steps/lmrescore_const_arpa.sh  $data_dir/lang_test_{tgsmall,fglarge} \
          $data_dir/test_clean $dec_dir $out_dir/decode_test_clean_fglarge   || exit 1;