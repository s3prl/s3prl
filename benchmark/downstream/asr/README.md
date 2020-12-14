# End-to-end Automatic Speech Recognition Systems - PyTorch Implementation
For complete introdution and usage, please see the original repository [Alexander-H-Liu/End-to-end-ASR-Pytorch](https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch).
## New features
1. SpecAugment
2. Label Smoothing
3. VGG encoder with Layer Normalization
4. Learning rate scheduler
5. Stabilizing joint acoustic and language model beam decoding by <eos> threshold


## Instructions
### Dataset
Download librispeech datset first in [OpenSLR website](http://www.openslr.org/12)
### Training
Modify `script/train.sh`, `script/train_lm.sh`, `config/librispeech_asr.yaml`, and `config/librispeech_lm.yaml` first. GPU is required.
```
bash script/train.sh <asr name> <cuda id>
bash script/train_lm.sh <lm name> <cuda id>
```
### Testing
Modify `script/test.sh` and `config/librispeech_test.sh` first. Increase the number of `--njobs` can speed up decoding process, but might cause OOM.
```
bash script/test.sh <asr name> <cuda id>
```

## LibriSpeech 100hr Result
This baseline is composed of a character-based joint CTC-attention ASR model and an RNNLM which were trained on the LibriSpeech `train-clean-100`. The perplexity of the LM on the `dev-clean` set is 2.79. 

| Decoding | DEV WER(%) | TEST WER(%) |
| -------- | ---------- | ----------- |
| Greedy   |   14.74    |    14.80    |
| B=2 + LM |   12.89    |    12.93    |
| B=4 + LM |   11.67    |    11.74    |
| B=8 + LM |   11.35    |    11.42    |



