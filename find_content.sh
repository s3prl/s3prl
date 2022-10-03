#!/bin/bash

if [ -z $1 ]; then
    echo "Usage: $0 <pattern>"
    exit 1
fi

ag $1 $(cat valid_paths.txt)
