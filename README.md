<div align="center">
    <br>
    <img src="https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning/blob/master/file/S3PRL-logo.png" width="900"/>
    <p>
    S3PRL: The Self-Supervised Speech Pre-training and Representation Learning Speech Toolkit 🦜, built on PyTorch, an 1-for-all interface for a wide variety of self-supervised upstream models and speech downstream tasks.
    </p>
    <hr/>
</div>
<p align="center">
    <a href="https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning/actions">
        <img alt="Build" src="https://github.com/allenai/allennlp/workflows/Master/badge.svg?event=push&branch=master">
    </a>
    <a href="https://en.wikipedia.org/wiki/MIT_License">
        <img alt="License" src="https://img.shields.io/github/license/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning">
    </a>
    <a href="#development-pattern-for-contributors">
        <img alt="Codecov" src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg">
    </a>
    <a href="https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning/issues">
        <img alt="Bitbucket open issues" src="https://img.shields.io/github/issues/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning">
    </a>
    <br/>
</p>

Updates
------------------------------------
- We are migrating to a newer version for a more general, flexible, and scalable code. See the introduction below for more information!
- The legacy verison can be accessed by checking out to the tag `v0.1.0`: `git checkout v0.1.0`.
- Any suggestions or pull requests are welcome!

------------------------------------

Introduction
------------------------------------
This is an open source project called S3PRL, which stands for **S**elf-**S**upervised **S**peech **P**re-training and **R**epresentation **L**earning. In this toolkit, various *upstream* self-supervised speech models are implemented with easy-to-load setups, and *downstream* evaluation tasks are available with easy-to-use scripts. Below is an intuitive illustration on how this toolkit may help you:

<img src="https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning/blob/master/file/S3PRL-interface.png" width="900"/>

- View the list of *upstreams* we support: [Upstream README](https://github.com/s3prl/s3prl/tree/master/upstream#upstream-models)
- View the list of *downstreams* we support: [Downstream README](https://github.com/s3prl/s3prl/tree/master/downstream#downstream-tasks)

Feel free to use or modify our toolkit in your research, any bug report or improvement suggestion will be appreciated. If you have any questions, please [open up a new issue](https://github.com/s3prl/s3prl/issues). If you find this toolkit helpful to your research, please do consider to cite [our papers](#Citation), thanks!

------------------------------------

Table of Contents
------------------------------------

<!--ts-->
   * [Table of contents](#table-of-contents)
   * [Installation](#installation)
   * [Data preparation](#data-preparation)
       * [Download extracted features (RECOMMENDED)](#download-extracted-features)
       * [Preprocessing with Librosa](#preprocessing-with-librosa)
       * [Preprocessing with Kaldi](#preprocessing-with-kaldi)
       * [On-the-fly Feature Extraction (RECOMMENDED)](#on-the-fly-feature-extraction)
       * [Downstream Task Preprocessing](#downstream-task-preprocessing)
   * [Train upstream models](#train-upstream-models)
       * [Train your own Mockingjay](#train-your-own-mockingjay)
       * [Train your own TERA](#train-your-own-tera)
       * [Train your own AALBERT](#train-your-own-aalbert)
   * [Downstream evaluations](#downstream-evaluations)
       * [Evaluating upstream models with phone classification](#evaluating-upstream-models-with-phone-classification)
       * [Evaluating upstream models with speaker recognition](#evaluating-upstream-models-with-speaker-recognition)
       * [Apply different knowledge transfer methods](#apply-different-knowledge-transfer-methods)
       * [Evaluating baseline features](#evaluating-baseline-features)
       * [Evaluating ASR with PyTorch-Kaldi scripts](#evaluating-asr-with-pytorch-kaldi-scripts)
   * [Evaluating your own model](#evaluating-your-own-model)
   * [Using upstream models with your own task](#using-upstream-models-with-your-own-task)
   * [Tutorial for application on custom dataset](#tutorial-for-application-on-custom-dataset)
   * [Supplementary Wiki Page](#supplementary-wiki-page)
       * [Extracting with Kaldi](#extracting-with-kaldi)
       * [ASR with PyTorch Kaldi](#asr-with-pytorch-kaldi)
   * [Development pattern for contributors](#development-pattern-for-contributors)
   * [Reference](#reference)
   * [Citation](#citation)
<!--te-->

------------------------------------

Installation
------------------------------------

- **Python** >= 3.6
- **PyTorch** version >= 1.7.0
- For training new models, you'll also need computing power (**high-end GPU**) and memory space (both RAM/GPU's RAM).
- To install fairseq and develop locally:
```bash=
git clone https://github.com/s3prl/s3prl.git
cd s3prl
pip install -r requirements.txt
```

[Back to Top](#table-of-contents)

------------------------------------
Data preparation
------------------------------------
### Download extracted features
- We provide the features we extracted for you to download directly: [S3PRL Drive](http://www.bit.ly/drive-S3PRL)
```bash
Structure of S3PRL Drive:
data/
    libri_mfcc_cmvn.zip 
    libri_fbank_cmvn.zip 
    libri_fmllr_cmvn.zip # features used for TERA
    timit_fmllr_cmvn.zip
    libri_mel160_subword5000 # features used for Mockingjay
```
- Download then unzip them, for example:
```bash
cd data/
unzip libri_fmllr_cmvn.zip
```
- Modify the setting in config files: [`config/downstream.yaml`](config/downstream.yaml), and others if needed:
```yaml
data_path: 'data/libri_fmllr_cmvn'
```

------------------------------------

### Preprocessing with Librosa
#### LibriSpeech
- Download the [LibriSpeech](http://www.openslr.org/12) dataset and place under [`data/`](data/): `data/LibriSpeech`. 
- The extracted data, which is ready for training, will be stored under the same [`data/`](data/) directory by default. 
```bash
# To preprocess different acoustic features, options are:
python preprocess/preprocess_libri.py --feature_type=mfcc --delta=True --delta_delta=True # this generates: /data/libri_mfcc39, window_size=25ms, stride=10ms
python preprocess/preprocess_libri.py --feature_type=fbank --delta=False # this generates: /data/libri_fbank80, window_size=25ms, stride=10ms
python preprocess/preprocess_libri.py --feature_type=fbank --delta=True # this generates: /data/libri_fbank160, window_size=25ms, stride=10ms
# features used for old Mockingjay pre-trained models (also for the Montreal phone set)
python preprocess/preprocess_libri.py --feature_type=linear --delta=False # 1025-dim, window_size=50ms, stride=12.5ms
python preprocess/preprocess_libri.py --feature_type=mel --delta=True # 160-dim, window_size=50ms, stride=12.5ms
```

#### TIMIT
- Download the [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1) dataset and place under [`data/`](data/): `data/timit`. 
- Follow the command used above:
```bash
python preprocess/preprocess_timit.py --feature_type=fbank --delta=False # 80-dim, window_size=25ms, stride=10ms
python preprocess/preprocess_timit.py --feature_type=mfcc --delta=True --delta_delta=True # 39-dim, window_size=25ms, stride=10ms
# old preprocessing settings:
python preprocess/preprocess_timit.py --feature_type=mel --data_path=../data/LibriSpeech # 160-dim, window_size=50ms, stride=12.5ms
python preprocess/preprocess_timit.py --feature_type=linear --delta=False # 1025-dim, window_size=50ms, stride=12.5ms
```

------------------------------------

### Preprocessing with Kaldi
- To extract with Kaldi, see the supplementary wiki page for detailed instructions: [Extracting with Kaldi](https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning/wiki/Extracting-with-Kaldi)
- Example codes are provided for the conversion of Kaldi .ark to .npy, which supports the format of a regular pytorch dataset.
    - TIMIT: [preprocess/ark2timit.py](https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning/blob/master/preprocess/ark2timit.py)
    - LibriSpeech: [preprocess/ark2libri.py](https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning/blob/master/preprocess/ark2libri.py)
    - VoxCeleb:  [preprocess/ark2voxceleb.py](https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning/blob/master/preprocess/ark2voxceleb.py)
- Or download the extracted features from here: [S3PRL Drive](http://www.bit.ly/drive-S3PRL)
- Place the downloaded `*.zip` files under [`data/`](data/):
```bash
cd data/
unzip libri_fmllr_cmvn.zip # features used for TERA
```

### On-the-fly Feature Extraction
- This feature allow users to run training and testing with out preprocessing data, feature extraction is done during runtime (This will not increase your training time!).
- To **enable bucketing** (optional, but substantially increase training efficiency), you need to run this script to get all the length of the training data.
```bash
python preprocess/generate_len_for_bucket.py --data_root=data/LibriSpeech/ # this generates: /data/len_for_bucket
```
Next change the following attribute in your `config/upstream.yaml` and `config/downstream.yaml`:
```yaml
dataloader:
    data_path: '/data/len_for_bucket'
```
- Finally, add the following argument when runing upstream/downstream scripts (pre-trained checkpoints will automatically use their saved `online.yaml` during pre-training, so no need to specify for pre-trained checkpoints):
```bash
--online_config=config/online.yaml
```

------------------------------------

### Downstream Task Preprocessing

#### Kaldi Phone Set (RECOMMENDED)
- 41 phone classes, this set is considered in the CPC, TERA papers.
- To use the CPC phone alignment data, use the following command:
```bash
cd data/cpc_phone
unzip converted_aligned_phones.zip
```
- Make sure that in [`config/downstream.yaml`](config/downstream.yaml), phone path is set to:
```yaml
phone_path: 'data/cpc_phone'
```
- ***IMPORTANT:** these phone alignments correspond to a feature/label for every 10ms, you need to use features with windows of 25 ms and an overlap of 10 ms, we recommend the [Kaldi extracted features](http://www.bit.ly/drive-S3PRL).*

#### Montreal Phone Set (for old Mockingjay pre-trained models)
- 72 phone classes, this set is considered in the Mockingjay paper.
- To use the [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/) phone alignment data, download the `libri_alignment.zip` from [S3PRL Drive](http://www.bit.ly/drive-S3PRL) and place under the [`data/`](data/) directory:
```bash
cd data
unzip libri_alignment.zip
cd ..
python preprocess/preprocess_alignment.py
```
- Change the setting in [`config/downstream.yaml`](config/downstream.yaml):
```yaml
phone_path: 'data/libri_phone'
```
- ***Warning:** you need to use `preprocess/preprocess_libri.py --feature_type=mel` to extract matching features.*

[Back to Top](#table-of-contents)

------------------------------------

Train upstream models
------------------------------------
- If you wish to train your own upstream models, 
please follow the instructions here: [Pretrain README](https://github.com/s3prl/s3prl/tree/master/pretrain)

[Back to Top](#table-of-contents)

------------------------------------
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

### Apply different knowledge transfer methods
#### Weighted sum from all layers:
- Simply add `--weighted_sum` to the above commands.
- For example, phone linear frame-wise classification on LibriSpeech:
```python
python run_downstream.py --weighted_sum --run=phone_linear --upstream=transformer --ckpt=path_to_ckpt/states-1000000.ckpt
```

#### Fine-tuning:
- Simply add `--fine_tune` to the above commands.
- For example, phone linear frame-wise classification on LibriSpeech:
```python
python run_downstream.py --fine_tune --run=phone_linear --upstream=transformer --ckpt=path_to_ckpt/states-1000000.ckpt
```

### Evaluating baseline features
- Simply change the `--upstream=transformer` to `--upstream=baseline`, and we no longer need to specify `--ckpt`.
- For example, phone linear frame-wise classification on LibriSpeech:
```python
python run_downstream.py --run=phone_linear --upstream=baseline
```

### Evaluating ASR with PyTorch-Kaldi scripts
- See the supplementary wiki page for detailed instructions: [ASR with PyTorch-Kalid](https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning/wiki/ASR-with-PyTorch-Kaldi)

[Back to Top](#table-of-contents)

------------------------------------

Evaluating your own model
------------------------------------
- You can easily insert your own upstream models to the evaluation script [`run_downstream.py`](run_downstream.py).
- There are only three simple requirements for each upstream model:
    1) Implement the `forward` method of `nn.Module`,
    2) Contains the `out_dim` attribute.
    3) Takes input and output in the shape of: (batch_size, time_steps, feature_dim)
- Initialize your model at the function `get_upstream_model` in [`run_downstream.py`](run_downstream.py):
```python
elif args.upstream == 'your_model':
    example_options = {'ckpt_file' : args.ckpt,
                       'input_dim' : args.input_dim,
                       'load_pretrain' : True}
    upstream_model = YOUR_MODEL(example_options)
```
- Now you can evaluate your model with `--upstream=your_model`.
- Make sure the input acoustic features align with your pre-trained model.

[Back to Top](#table-of-contents)

------------------------------------

Using upstream models with your own task
------------------------------------
- You can also fine-tune or extract from the pre-trained upstream model on your own dataset and tasks! 
- ***IMPORTANT:** 
  You must use input acoustic features with the **same preprocessing settings and pipeline** as pre-trained models!!!*
- Pre-trained checkpoints can be download from: [S3PRL Drive](http://www.bit.ly/drive-S3PRL)
    - *Mockingjay Models:* 
    Download the data of `libri_mel160_subword5000.zip`, or follow the pipeline used in `python preprocess/preprocess_libri.py --feature_type=mel` to extract identical ***160-dim mel*** features.
    - *TERA Models:* 
    Download the data of `libri_fmllr_cmvn.zip`, or follow the pipeline used in the *Kaldi s5 recipe* to extract identical ***40-dim fmllr*** features.
    - *AALBERT Models:* 
    Coming soon, download the data of `libri_fbank_cmvn.zip`, or follow the pipeline used in the *Kaldi s5 recipe* to extract identical ***80-dim fbank*** features.
- ***WARNING:** 
  If you are getting bad or worse results, it's probably caused by the **mismatch of acoustic features** between pre-trained models and downstream task!!!* 
- Below we show an [example code](src/example_extract_finetune.py) of fine-tuning an upstream model with your own downstream model, by using the wrapper class in [nn_transformer.py](transformer/nn_transformer.py):
```python
import torch
from transformer.nn_transformer import TRANSFORMER
from downstream.model import example_classifier
from downstream.solver import get_optimizer

# setup the transformer model
"""
`options`: a python dictionary containing the following keys:
    ckpt_file: str, a path specifying the pre-trained ckpt file
    load_pretrain: str, ['True', 'False'], whether to load pre-trained weights
    no_grad: str, ['True', 'False'], whether to have gradient flow over this class
    dropout: float/str, use float to modify dropout value during downstream finetune, or use the str `default` for pre-train default values
    spec_aug: str, ['True', 'False'], whether to apply SpecAugment on inputs (used for ASR training)
    spec_aug_prev: str, ['True', 'False'], apply spec augment on input acoustic features if True, else apply on output representations (used for ASR training)
    weighted_sum: str, ['True', 'False'], whether to use a learnable weighted sum to integrate hidden representations from all layers, if False then use the last
    select_layer: int, select from all hidden representations, set to -1 to select the last (will only be used when weighted_sum is False)
    permute_input: str, ['True', 'False'], this attribute is for the forward method. If Ture then input ouput is in the shape of (T, B, D), if False then in (B, T, D)
"""
options = {
    'ckpt_file'     : './result/result_transformer/tera/fmllrBase960-F-N-K-libri/states-1000000.ckpt',
    'load_pretrain' : 'True',
    'no_grad'       : 'True',
    'dropout'       : 'default',
    'spec_aug'      : 'False',
    'spec_aug_prev' : 'True',
    'weighted_sum'  : 'False',
    'select_layer'  : -1,
    'permute_input' : 'False',
}
transformer = TRANSFORMER(options=options, inp_dim=0) # set `inpu_dim=0` to auto load the `inp_dim` from `ckpt_file`

# setup your downstream class model
classifier = example_classifier(input_dim=768, hidden_dim=128, class_num=2).cuda()

# construct the optimizer
params = list(transformer.named_parameters()) + list(classifier.named_parameters())
optimizer = get_optimizer(params=params, lr=4e-3, warmup_proportion=0.7, training_steps=50000)

# forward
example_inputs = torch.zeros(3, 1200, 40) # A batch of spectrograms:  (batch_size, time_step, feature_size)
# IMPORTANT: Input acoustic features must align with the ones used during our pre-training!
reps = transformer(example_inputs) # returns: (batch_size, time_step, feature_size)
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

[Back to Top](#table-of-contents)

Tutorial for application on custom dataset
------------------------------------
For any arbitrary dataset that looks like this:
```
- Custom_dataset/
    - Custom_train/
       - *.wav / flac / mp3 ...
    - Custom_dev/
       - *.wav / flac / mp3 ...
    - Custom_test/
       - *.wav / flac / mp3 ...
```
The script `preprocess/preprocess_any.py` will process the "train", "dev", "test" set one by one:
```bash
python preprocess/preprocess_any.py --audio_extention=.flac
```

Users only need to specify the path of the directory of each set.
So for the example above: 
- the path to the "train" set should be: `Custom_dataset/Custom_train/`
- the path to the "dev" set should be: `Custom_dataset/Custom_dev/`
- the path to the "test" set should be: `Custom_dataset/Custom_test/`

The generated files will be compatible to our dataloader.


Also, in your config file `*.yaml`, these should be changed:
```yaml
  data_path: 'data/NewData_fbank80' 
  train_set: ['train']
  dev_set: ['dev'] 
  test_set: ['test']
```

[Back to Top](#table-of-contents)

Supplementary Wiki Page
------------------------------------
#### [Extracting with Kaldi](https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning/wiki/Extracting-with-Kaldi)
#### [ASR with PyTorch Kaldi](https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning/wiki/ASR-with-PyTorch-Kaldi)

[Back to Top](#table-of-contents)

Development pattern for contributors
------------------------------------
1. [Create a personal fork](https://help.github.com/articles/fork-a-repo/) of the [main S3PRL repository](https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning) in GitHub.
2. Make your changes in a named branch different from `master`, e.g. you create a branch `new-awesome-feature`.
3. [Generate a pull request](https://help.github.com/articles/creating-a-pull-request/) through the Web interface of GitHub.
4. Please verify that your code is free of basic mistakes, we appreciate any contribution!

[Back to Top](#table-of-contents)

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

[Back to Top](#table-of-contents)

Citation
------------------------------------
- The S3PRL Toolkit:
```
@misc{S3PRL,
  author = {Andy T. Liu and Yang Shu-wen},
  title = {S3PRL: The Self-Supervised Speech Pre-training and Representation Learning Toolkit},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning}
}
```

Here we also list all papers that use our toolkit (Feel free to add your own paper by making a pull request).
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
@misc{tera,
    title={TERA: Self-Supervised Learning of Transformer Encoder Representation for Speech},
    author={Andy T. Liu and Shang-Wen Li and Hung-yi Lee},
    year={2020},
    eprint={2007.06028},
    archivePrefix={arXiv},
    primaryClass={eess.AS}
}
```
- Mockingjay for Adversarial Defense, code for computing LNSR: [utility/observe_lnsr.py](https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning/blob/master/utility/observe_lnsr.py)
```
@misc{mockingjay_defense,
    title={Defense for Black-box Attacks on Anti-spoofing Models by Self-Supervised Learning},
    author={Haibin Wu and Andy T. Liu and Hung-yi Lee},
    year={2020},
    eprint={2006.03214},
    archivePrefix={arXiv},
    primaryClass={eess.AS}
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
