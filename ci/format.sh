#!/bin/bash

# set -x
set -e

VALID_ARGS=$(getopt -o ch --long check,help -- "$@")
if [[ $? -ne 0 ]]; then
    exit 1;
fi

eval set -- "$VALID_ARGS"
check_only=false;
while [ : ]; do
  case "$1" in
    -h | --help)
        echo "Usage: $0 [--check] [file1, file2, file3 ...]"
        echo "If no file is given, use the files under ./valid_paths.txt"
        echo "if --check, do not apply changes on the files"
        exit 0
        ;;
    -c | --check)
        shift
        check_only=true;
        ;;
    --) shift;
        break
        ;;
  esac
done

files="$*"
if [ -z "$files" ]; then
    script_dir=$(dirname $0)
    script_dir=$(realpath $script_dir)
    s3prl_dir=$(dirname $script_dir)
    files=$(cat ${s3prl_dir}/valid_paths.txt)
fi
echo "Formatting the files: $files"

echo "Run flake8"
# stop the build if there are Python syntax errors or undefined names
flake8 $files --count --select=E9,F63,F7,F82 --show-source --statistics

# exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
flake8 $files --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
echo "flake8 successes"

echo "Run black"
if [ "$check_only" == true ]; then
    black --check $files
else
    black $files
fi
echo "black successes"

echo "Run isort"
third_party=$(cat $(ls requirements/*) | cut -d " " -f 1)
if [ "$check_only" == true ]; then
    isort --profile black --thirdparty "$third_party" --check $files
else
    isort --profile black --thirdparty "$third_party" $files
fi
echo "isort successes"

if [ "$check_only" == true ]; then
    echo "Successfully pass the format check!"
fi
