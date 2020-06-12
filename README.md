# 🦜 The S3PRL Toolkit
## The Self-Supervised Speech Pre-training and Representation Learning Toolkit
### PyTorch Official Implementation
[![GitHub](https://img.shields.io/github/license/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning)](https://en.wikipedia.org/wiki/MIT_License)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Bitbucket open issues](https://img.shields.io/bitbucket/issues/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning)](https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning/issues)

## Introduction
This is an open source project called S3PRL, where various self-supervised algorithms of speech are implemented:
- **Mockingjay** -　["Mockingjay: Unsupervised Speech Representation Learning with Deep Bidirectional Transformer Encoders"](https://arxiv.org/abs/1910.12638), Oral Lecture in [ICASSP 2020](https://2020.ieeeicassp.org/).
- **TERA** - ["TERA: Self-Supervised Pre-training of Transformer Encoders for Speech"](), codes and paper are coming soon. #TODO
- **Audio ALBERT** - ["Audio ALBERT: A Lite BERT for Self-supervised Learning of Audio Representation"](https://arxiv.org/abs/2005.08575), Submitted to [INTERSPEECH 2020](http://www.interspeech2020.org/), codes are coming soon. #TODO
- **APC** - ["An Unsupervised Autoregressive Model for Speech Representation Learning"](https://arxiv.org/abs/1904.03240), Accepted by [INTERSPEECH 2019](https://interspeech2019.org/).

With standard downstream evaluation scripts available:
- **Phone classification:** *Linear* classifiers and *1 Hidden* classifiers, with 41 phone classes on LibriSpeech `train-clean-100` with fixed train/test splits, as proposed and used in the [CPC](https://arxiv.org/abs/1807.03748) and [TERA]() paper. #TODO
- **Speaker recognition:** *Frame-wise* and *Utterance-wise* linear classifiers, with 251 speaker classes on LibriSpeech `train-clean-100` with fixed train/test splits, as proposed and used in the [CPC](https://arxiv.org/abs/1807.03748), [AALBERT](https://arxiv.org/abs/2005.08575) and [TERA]() paper. #TODO
- **ASR speech recognition:** *DNN/HMM hybrid* speech recognition systems with the [PyTorch-Kaldi Toolkit](https://github.com/mravanelli/pytorch-kaldi), training and evaluating scripts are coming soon. #TODO
- **Sentiment classification on spoken content:** simple *one-layer RNN* classifier on MOSEI dataset, as proposed and used in [Mockingjay](https://arxiv.org/abs/1910.12638)

Feel free to use or modify them, any bug report or improvement suggestion will be appreciated. If you have any questions, please contact tingweiandyliu@gmail.com. If you find this project helpful for your research, please do consider to cite [our papers](#Citation), thanks!

## Prerequisite
- #TODO

## Data Preprocessing
- #TODO

## Train your own model
- #TODO

## Downstream Evaluation
- #TODO

## Using pre-trained models with your own tasks
- #TODO

## Reference
1. [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/), McAuliffe et. al.
2. [CMU MultimodalSDK](https://github.com/A2Zadeh/CMU-MultimodalSDK/blob/master/README.md), Amir Zadeh.
3. [PyTorch Transformers](https://github.com/huggingface/pytorch-transformers), Hugging Face.
4. [Autoregressive Predictive Coding](https://arxiv.org/abs/1904.03240), Yu-An Chung.
5. [Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748), Aaron van den Oord.
5. [End-to-end ASR Pytorch](https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch), Alexander-H-Liu.
6. [Tacotron Preprocessing](https://github.com/r9y9/tacotron_pytorch), Ryuichi Yamamoto (r9y9)

## Citation
- Mockingjay:
```
@article{Liu_2020,
   title={Mockingjay: Unsupervised Speech Representation Learning with Deep Bidirectional Transformer Encoders},
   ISBN={9781509066315},
   url={http://dx.doi.org/10.1109/ICASSP40776.2020.9054458},
   DOI={10.1109/icassp40776.2020.9054458},
   journal={ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
   publisher={IEEE},
   author={Liu, Andy T. and Yang, Shu-wen and Chi, Po-Han and Hsu, Po-chun and Lee, Hung-yi},
   year={2020},
   month={May}
}
```

- Understanding SAT:
```
@misc{yang2020understanding,
    title={Understanding Self-Attention of Self-Supervised Audio Transformers},
    author={Shu-wen Yang and Andy T. Liu and Hung-yi Lee},
    year={2020},
    eprint={2006.03265},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
