#!/bin/bash

set -e
set -x

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <expdir1> <expdir2> ..."
  echo "e.g., ./downstream/sv_voxceleb1/test_expdirs.sh result/downstream/exp1 result/downstream/exp2"
  exit 1
fi

for expdir in $*;
do
    if [ -d $expdir ]; then
        ./downstream/sv_voxceleb1/test_expdir.sh $expdir python3 run_downstream.py -m evaluate
    fi
done

