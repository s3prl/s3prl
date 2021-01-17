#!/bin/bash

for i in $(seq 1 100); do
    if [ -f $expdir"/train_finished" ]; then
        break
    fi
    commands="$*"
    eval $commands
done

