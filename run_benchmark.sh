#!/bin/bash

# start a new training experiment
python3 run_benchmark.py -m train -c benchmark/downstream/example/config.yaml -d example -u example -n HelloWorld

# resume a checkpoint
python3 run_benchmark.py -m train -e [ckpt]

# test a checkpoint
python3 run_benchmark.py -m evaluate -e [ckpt]
