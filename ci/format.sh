#!/bin/bash

set -x
set -e

pip install flake8 black

script_dir=$(dirname $0)
script_dir=$(readlink -f $script_dir)
s3prl_dir=$(dirname $script_dir)

# stop the build if there are Python syntax errors or undefined names
flake8 $(cat ${s3prl_dir}/valid_paths.txt) --count --select=E9,F63,F7,F82 --show-source --statistics

# exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
flake8 $(cat ${s3prl_dir}/valid_paths.txt) --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# black
black --check $(cat ${s3prl_dir}/valid_paths.txt)
