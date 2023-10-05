#!/bin/bash

python3 s3prl/main.py SuperbASR --target_dir result/asr --prepare_data.dataset_root /home/leo/d/datasets/LibriSpeech/ --build_upstream.name apc
