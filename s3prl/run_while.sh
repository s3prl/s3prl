#!/bin/bash

command="$*"
for i in $(seq 1 100); do
    uuid=$(uuidgen)
    message="[run_while] job ${uuid} for trial ${i}"
    
    unique_command="echo ${message}; $command;"
    eval "$unique_command"
    
    pkill -f "${message}"
done
