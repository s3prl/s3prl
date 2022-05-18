#!/bin/bash

set -x
set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 [check_only_or_not (true/false)] [file1] [file2] ..."
    exit 1
fi
check_only="$1"
shift

script_dir=$(dirname $0)
script_dir=$(realpath $script_dir)
s3prl_dir=$(dirname $script_dir)

files="$*"
if [ -z "$files" ]; then
    files=$(cat ${s3prl_dir}/valid_paths.txt)
fi

# stop the build if there are Python syntax errors or undefined names
flake8 $files --count --select=E9,F63,F7,F82 --show-source --statistics

# exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
flake8 $files --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# black
if [ "$check_only" == true ]; then
    black --check $files
else
    black $files
fi

# isort
if [ "$check_only" == true ]; then
    isort --profile black --check $files
else
    isort --profile black $files
fi

if [ "$check_only" == true ]; then
    echo "Successfully pass format check!"
fi
