#!/bin/bash

set -e

if [ "$#" != 2 ]; then
    echo Usage. $0 expdir voxceleb1_root
fi

expdir=$1
voxceleb1=$2

if [ ! -d "$expdir" ]; then
    echo "The expdir does not exist!"
    exit 1
fi

if [ ! -d "$voxceleb1" ]; then
    echo "VoxCeleb1 dataset does not exist!"
    exit 1
fi

echo "Start testing ckpts..."
for state_name in 20000 40000 60000 80000 100000 120000 140000 160000 180000 200000;
do
    ckpt_path="$expdir/states-$state_name.ckpt"
    echo "Testing $ckpt_path"
    if [ ! -f "$ckpt_path" ]; then
        continue
    fi

    log_dir="$expdir/states-$state_name"
    if [ ! -d "$log_dir" ] || [ "$(cat "$log_dir"/log.txt | grep "test-EER" | wc -l)" -lt 1 ] || [ ! -f $log_dir/test_predict.txt ]; then
        mkdir -p $log_dir
        override=args.expdir=${log_dir},,config.downstream_expert.datarc.file_path=${voxceleb1}
        python3 run_downstream.py -m evaluate -e $ckpt_path -o $override > $log_dir/log.txt
    fi
done

echo "Report the testing results..."
report=$expdir/report.txt
grep test-EER $expdir/*/log.txt | sort -nrk 2 > $report
ckpt_num=$(cat $report | wc -l)
cat $report

echo
echo "$ckpt_num checkpoints evaluated."
echo "The best checkpoint achieves EER $(cat $report | tail -n 1 | cut -d " " -f 2)"

echo
echo "Prepare prediction file for submission..."
best_prediction=$(realpath $(dirname $(cat $report | tail -n 1 | cut -d ":" -f 1))/test_predict.txt)

target=$expdir/test_predict.txt
if [ -f $target ]; then
    rm $target
fi
target=$(realpath $target)
ln -s $best_prediction $target

echo "The best prediction file has been prepared"
echo "${best_prediction} -> ${target}"
