#!/bin/bash

rm -rf ./source/s3prl.*
rm -rf ./source/_autosummary

make clean html

touch build/html/.nojekyll

