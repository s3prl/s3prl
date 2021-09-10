#!/bin/bash

output_dir=$1
if [ -z $output_dir ]; then
    echo [Usage] $0 [output_dir]
    exit 1
fi
mkdir -p $output_dir

url="https://superbbenchmark.org/api/download/expdirs"
wget $url -O $output_dir/expdirs.zip

cd $output_dir
unzip -q expdirs.zip
cd ../
expdirs=$output_dir/expdirs
echo "[Expdirs Downloaded] ${expdirs} contains the minimum files necessary for a submission which should appear in each task's expdir after Testing."

python3 submit.py --output_dir $output_dir \
    --pr $expdirs/pr_expdir \
    --sid $expdirs/sid_expdir \
    --ks $expdirs/ks_expdir \
    --ic $expdirs/ic_expdir \
    --er_fold1 $expdirs/er_fold1_expdir \
    --er_fold2 $expdirs/er_fold2_expdir \
    --er_fold3 $expdirs/er_fold3_expdir \
    --er_fold4 $expdirs/er_fold4_expdir \
    --er_fold5 $expdirs/er_fold5_expdir \
    --asr_no_lm $expdirs/asr_expdir \
    --asr_with_lm $expdirs/asr_expdir \
    --qbe $expdirs/qbe_expdir \
    --sf $expdirs/sf_expdir \
    --sv $expdirs/sv_expdir \
    --sd $expdirs/sd_expdir

echo "[Complete] ${output_dir}/predict.zip is the file you should submit :)"
