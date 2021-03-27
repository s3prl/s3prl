#!/bin/bash

set -x
set -e

for i in $(seq 1 100); do
    eval "$*"
done

