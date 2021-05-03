#!/bin/bash

expdir=$1
shift
if [ ! -d "$expdir" ]; then
    echo "The expdir does not exits!"
    exit 1
fi

ckpt_num=$(grep test-ERR $expdir/states-*.result | sort -nrk 2 | tee /dev/tty | wc -l)
echo $ckpt_num checkpoints evaluated.
