#!/bin/bash

# start a new training experiment
python3 run_benchmark.py --mode train --config benchmark/downstream/example/config.yaml --downstream example --upstream example --upstream_trainable --expname HelloWorld

# resume a checkpoint
python3 run_benchmark.py --mode train --past_exp [ckpt]

# test a checkpoint
python3 run_benchmark.py --mode evaluate --past_exp [ckpt]
