#!/bin/bash

set -e

conda="/home/leo/miniconda3/etc/profile.d/conda.sh"
source "$conda"

env1=$1
shift

env2=$1
shift

commands="$*"

echo "Running with conda env: $env1"
conda activate $env1
eval "$commands -o $env1.pth"
conda deactivate

echo "Running with conda env: $env2"
conda activate $env2
eval "$commands -o $env2.pth"
conda deactivate

python3 allclose.py "$env1.pth" "$env2.pth"

