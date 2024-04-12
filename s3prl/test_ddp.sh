#!/bin/bash

gpus=2;
distributed="-m torch.distributed.launch --nproc_per_node ${gpus}";
python3 $distributed run_downstream.py -m train -n testing_ddp -u fbank -d example \
        -o config.runner.gradient_accumulate_steps=2
