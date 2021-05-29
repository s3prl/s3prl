#!/bin/bash

set -x
set -e

report=report.log

if [ -f "$report" ]; then
    rm "$report"
fi

for upstream in fbank cpc apc vq_apc tera wav2vec vq_wav2vec wav2vec2;
do
    python3 test_env.py -m "test" -u "$upstream" --report "$report" --refresh
done

cat "$report" | sort -nk 1 