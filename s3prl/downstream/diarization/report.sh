#!/bin/bash

set -e
set -x

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <expdir>"
  echo "e.g., ./downstream/diarization/report.sh result/downstream/test"
  exit 1
fi

scoring_dir="$1/scoring"
grep OVER $scoring_dir/result_th0.[^_]*_med[^_]* | sort -nrk 4
