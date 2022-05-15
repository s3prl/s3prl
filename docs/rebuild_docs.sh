#!/bin/bash

./clean_docs.sh

ignore_paths=$(python3 util/is_valid.py ../s3prl ../valid_paths.txt)

cd ../
sphinx-apidoc -d 3 --separate --implicit-namespace -o ./docs/source s3prl $ignore_paths
cd docs

make clean
make html

