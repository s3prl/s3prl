#!/bin/bash

set -e
set -x

expdir=$1
shift
if [ ! -d "$expdir" ]; then
    echo "The expdir does not exits!"
    exit 1
fi

commands=$*
if [ -z "$commands" ]; then
    echo "You should specify the evaluation command except assigning -e"
fi

echo Start evaluaing ckpts...
for state_name in 20000 40000 60000 80000 100000 120000 140000 160000 180000 200000;
do
    ckpt_path="$expdir/states-$state_name.ckpt"
    if [ ! -f "$ckpt_path" ]; then
        continue
    fi

    log_path="$expdir/states-$state_name.result"
    if [ ! -f "$log_path" ] || [ "$(cat "$log_path" | grep "test-ERR" | wc -l)" -lt 1 ]; then
        eval "$commands -e $ckpt_path" > $log_path
    fi
done
