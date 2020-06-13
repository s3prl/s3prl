# 🦜 The S3PRL Toolkit
- The **S**elf-**S**upervised **S**peech **P**re-training and **R**epresentation **L**earning Toolkit
- Official Implementation in PyTorch

[![GitHub](https://img.shields.io/github/license/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning)](https://en.wikipedia.org/wiki/MIT_License)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Bitbucket open issues](https://img.shields.io/bitbucket/issues/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning)](https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning/issues)

Introduction
------------------------------------
This is an open source project called S3PRL, where various *upstream* self-supervised speech models are implemented, and *downstream* evaluation tasks are available with easy-to-use scripts.

**Upstream Models:**
- **Mockingjay**
    - ["Mockingjay: Unsupervised Speech Representation Learning with Deep Bidirectional Transformer Encoders"](https://arxiv.org/abs/1910.12638)
    - Transformer based, BERT-style masked reconstruction loss
    - Oral Lecture in [ICASSP 2020](https://2020.ieeeicassp.org/).
- **TERA**
    - ["TERA: Self-Supervised Pre-training of Transformer Encoders for Speech"]()
    - Transformer based, multi-target alteration reconstruction loss
    - Codes and paper are coming soon. #TODO
- **Audio ALBERT**
    - ["Audio ALBERT: A Lite BERT for Self-supervised Learning of Audio Representation"](https://arxiv.org/abs/2005.08575)
    - Transformer based, BERT-style masked reconstruction loss
    - Submitted to [INTERSPEECH 2020](http://www.interspeech2020.org/), codes are coming soon. #TODO
- **APC**
    - ["An Unsupervised Autoregressive Model for Speech Representation Learning"](https://arxiv.org/abs/1904.03240)
    - RNN based, unidirectional reconstruction loss
    - Accepted by [INTERSPEECH 2019](https://interspeech2019.org/).

**Downstream Tasks:**
- **Phone classification:** 
    - *Linear* classifiers
    - *1 Hidden* classifiers
    - *Concat* classifiers
    - 41 phone classes on LibriSpeech `train-clean-100` with fixed train/test splits
    - Proposed and used in the [CPC](https://arxiv.org/abs/1807.03748) and [TERA]() paper.
- **Speaker recognition:** 
    - *Frame-wise* linear classifier
    - *Utterance-wise* linear classifier
    - 251 speaker classes on LibriSpeech `train-clean-100` with fixed train/test splits
    - Proposed and used in the [CPC](https://arxiv.org/abs/1807.03748), [AALBERT](https://arxiv.org/abs/2005.08575) and [TERA]() paper.
- **ASR speech recognition:** 
    - *DNN/HMM hybrid* speech recognition systems with the [PyTorch-Kaldi Toolkit](https://github.com/mravanelli/pytorch-kaldi)
    - Training and evaluating scripts are coming soon. #TODO
- **Sentiment classification on spoken content:** 
    - simple *one-layer RNN* classifier on MOSEI dataset
    - Proposed and used in [Mockingjay](https://arxiv.org/abs/1910.12638).

**Usage Highlight:**
- **Acoustic feature extraction scripts:**
    - Pre-processing with [Lirbosa](https://librosa.github.io/librosa/): *mfcc, fbank, mel*
    - Pre-processing with [TTS-Preprocessing](https://github.com/r9y9/tacotron_pytorch): *mel, linear*
    - Pre-processing with the [Kaldi](https://github.com/kaldi-asr/kaldi) s5 recipe: *mfcc, fbank, fmllr*
    - see section: *Data preporation*
- **Pre-train your own self-supervised models:**
    - Implementation of various upstream algorithms.
    - Pre-train them on your own data.
    - see section: *Train upstream models*
- **Evaluate your own pre-trained model:**
    - Easy-to-use downstream evaluation scripts.
    - Incorporate any pre-trained model of your own.
    - see section: *Evaluating your own model*
- **Apply pre-trained models on your own task:**
    - Easy-to-use pre-trained model initialization.
    - Incorporate any downstream task with the provided pre-trained models.
    - Implemented as [PyTorch-Kaldi](https://github.com/mravanelli/pytorch-kaldi) ready DNNs.
    - see section: *Using upstream models with your own task*

Feel free to use or modify them, any bug report or improvement suggestion will be appreciated. If you have any questions, please contact tingweiandyliu@gmail.com. If you find this project helpful for your research, please do consider to cite [our papers](#Citation), thanks!

Prerequisite
------------------------------------
- #TODO

Data preporation
------------------------------------
- #TODO

Train upstream models
------------------------------------
- #TODO

Downstream evaluations
------------------------------------
- The below commands are used for evaluating the transformer models, where we specify `--upstream=transformer`.
- The type of pre-trained transformers (Mockingjay, AALBERT, TERA) will be decided by the pre-trained checkpoint: `--ckpt`.

### Evaluating upstream models with phone classification
```python
# **Phone Linear** Frame-wise Classification on LibriSpeech
python run_downstream.py --run=phone_linear --upstream=transformer --ckpt=path_to_ckpt/states-1000000.ckpt

# **Phone 1 Hidden** Frame-wise Classification on LibriSpeech
python run_downstream.py --run=phone_1hidden --upstream=transformer --ckpt=path_to_ckpt/states-1000000.ckpt

# **Phone Concat** Frame-wise Classification on LibriSpeech
python run_downstream.py --run=phone_concat --upstream=transformer --ckpt=path_to_ckpt/states-1000000.ckpt
```

### Evaluating upstream models with speaker recognition
```python
# **Speaker Frame**-wise Classification on LibriSpeech
python run_downstream.py --run=speaker_frame --upstream=transformer --ckpt=path_to_ckpt/states-1000000.ckpt

# **Speaker Utterance**-wise Classification on LibriSpeech
python run_downstream.py --run=speaker_utterance --upstream=transformer --ckpt=path_to_ckpt/states-1000000.ckpt
```

### Evaluating baseline features
- simply change the `--upstream=transformer` to `--upstream=baseline`, and we no longer need to specify `--ckpt`.
- for example, phone linear frame-wise classification on LibriSpeech:
```python
python run_downstream.py --run=phone_linear --upstream=baseline
```

Evaluating your own model
------------------------------------
- You can easily insert your own upstream models to the evaluation script `run_downstream.py`.
- There are only three simple requirements for each upstream model:
    1) Implement the `forward` method of `nn.Module`,
    2) Contains the `out_dim` attribute.
    3) Takes input and output in the shape of: (batch_size, time_steps, feature_dim)
- Initialize your model in `get_upstream_model` of `run_downstream.py`:
```python
elif args.upstream == 'your_model':
    example_options = {'ckpt_file' : args.ckpt,
                       'input_dim' : args.input_dim,
                       'load_pretrain' : True}
    upstream_model = YOUR_MODEL(example_options)
```
- Now you can evaluate your model with `--upstream=your_model`.
- Make sure the input acoustic features align with your pre-trained model.

Using upstream models with your own task
------------------------------------
- You can also fine-tune or extract from the pre-trained upstream model on your own dataset and tasks! 
- *important: you must use input acoustic features with the **same preprocessing settings and pipeline** as pre-trained models!!!* 
- Below we show an [example code](src/example_extract_finetune.py) of fine-tuning a upstream model with your own downstream model, by using the wrapper class in [nn_transformer.py](transformer/nn_transformer.py):
```python
import torch
from transformer.nn_transformer import TRANSFORMER
from downstream.model import example_classifier
from downstream.solver import get_optimizer

# setup the transformer model
options = {
    'ckpt_file'     : './result/result_transformer/tera/fmllrBase960-F-N-K-libri/states-1000000.ckpt',
    'load_pretrain' : 'True',
    'no_grad'       : 'True',
    'dropout'       : 'default',
    'spec_aug'      : 'False',
    'spec_aug_prev' : 'True',
    'weighted_sum'  : 'False',
    'select_layer'  : -1,
}
transformer = TRANSFORMER(options=options, inp_dim=40)

# setup your downstream class model
classifier = example_classifier(input_dim=768, hidden_dim=128, class_num=2).cuda()

# construct the optimizer
params = list(transformer.named_parameters()) + list(classifier.named_parameters())
optimizer = get_optimizer(params=params, lr=4e-3, warmup_proportion=0.7, training_steps=50000)

# forward
example_inputs = torch.zeros(1200, 3, 40) # A batch of spectrograms: (time_step, batch_size, dimension)
reps = transformer(example_inputs) # returns: (time_step, batch_size, hidden_size)
reps = reps.permute(1, 0, 2) # change to: (batch_size, time_step, feature_size)
labels = torch.LongTensor([0, 1, 0]).cuda()
loss = classifier(reps, labels)

# update
loss.backward()
optimizer.step()

# save
PATH_TO_SAVE_YOUR_MODEL = 'example.ckpt'
states = {'Classifier': classifier.state_dict(), 'Transformer': transformer.state_dict()}
# torch.save(states, PATH_TO_SAVE_YOUR_MODEL)
```

Development pattern for contributors
------------------------------------
1. [Create a personal fork](https://help.github.com/articles/fork-a-repo/) of the [main S3PRL repository](https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning) in GitHub.
2. Make your changes in a named branch different from `master`, e.g. you create a branch `new-awesome-feature`.
3. [Generate a pull request](https://help.github.com/articles/creating-a-pull-request/) through the Web interface of GitHub.
4. Please verify that your code is free of basic mistakes, we appreciate any contribution!
   
Reference
------------------------------------
1. [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/), McAuliffe et. al.
2. [CMU MultimodalSDK](https://github.com/A2Zadeh/CMU-MultimodalSDK/blob/master/README.md), Amir Zadeh.
3. [PyTorch Transformers](https://github.com/huggingface/transformers), Hugging Face.
4. [Autoregressive Predictive Coding](https://github.com/iamyuanchung/Autoregressive-Predictive-Coding), Yu-An Chung.
5. [Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748), Aaron van den Oord.
5. [End-to-end ASR Pytorch](https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch), Alexander-H-Liu.
6. [Tacotron Preprocessing](https://github.com/r9y9/tacotron_pytorch), Ryuichi Yamamoto (r9y9)
7. [PyTorch-Kaldi](https://github.com/mravanelli/pytorch-kaldi), Mirco Ravanelli
8. [Kaldi](https://github.com/kaldi-asr/kaldi), Kaldi-ASR

Citation
------------------------------------
- Mockingjay:
```
@article{mockingjay,
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
- TERA:
```
@article{coming2020soon
}
```
- Understanding SAT:
```
@misc{understandingSAT,
    title={Understanding Self-Attention of Self-Supervised Audio Transformers},
    author={Shu-wen Yang and Andy T. Liu and Hung-yi Lee},
    year={2020},
    eprint={2006.03265},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
