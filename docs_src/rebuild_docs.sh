#!/bin/bash

rm -rf ./source/s3prl.*
rm -rf ./source/_autosummary

make clean html

rm -rf ../docs/
mkdir -p ../docs/
cp -r build/html/* ../docs/
touch ../docs/.nojekyll

