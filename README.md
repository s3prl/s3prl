# 🦜 Mockingjay
## Unsupervised Speech Representation Learning with Deep Bidirectional Transformer Encoders
### PyTorch Official Implementation
[![Bitbucket open issues](https://img.shields.io/bitbucket/issues/andi611/Mockingjay-Speech-Representation)](https://github.com/andi611/Mockingjay-Speech-Representation/issues)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![GitHub](https://img.shields.io/github/license/andi611/Mockingjay-Speech-Representation)](https://en.wikipedia.org/wiki/MIT_License)

This is an open source project for Mockingjay, an unsupervised algorithm for learning speech representations introduced and described in the paper ["Mockingjay: Unsupervised Speech Representation Learning with Deep Bidirectional Transformer Encoders"](https://arxiv.org/abs/1910.12638).
<img src="https://github.com/andi611/Mockingjay-Speech-Representation/blob/master/paper/training.png">

Feel free to use or modify them, any bug report or improvement suggestion will be appreciated. If you have any questions, please contact r07942089@ntu.edu.tw. If you find this project helpful for your research, please do consider to cite [this paper](#Citation), thanks!

# Highlight
## Pre-trained Models
You can find pre-trained models here:

 **[http://bit.ly/result_mockingjay](http://bit.ly/result_mockingjay)**

 Their usage are explained bellow and furthur in [Step 3 of the Instruction Section](#Instructions).

## Extracting Speech Representations
With this repo and the trained models, you can use it to extract speech representations from your target dataset. To do so, feed-forward the trained model on the target dataset and retrieve the extracted features by running the following example python code ([example_extract.py](example_extract.py)):
```python
import torch
from runner_mockingjay import get_mockingjay_model

example_path = 'result/result_mockingjay/mockingjay_libri_sd1337_LinearLarge/mockingjay-500000.ckpt'
mockingjay = get_mockingjay_model(from_path=example_path)

# A batch of spectrograms: (batch_size, seq_len, hidden_size)
spec = torch.zeros(3, 800, 160)

# reps.shape: (batch_size, num_hiddem_layers, seq_len, hidden_size)
reps = mockingjay.forward(spec=spec, all_layers=True, tile=True)

# reps.shape: (batch_size, num_hiddem_layers, seq_len // downsample_rate, hidden_size)
reps = mockingjay.forward(spec=spec, all_layers=True, tile=False)

# reps.shape: (batch_size, seq_len, hidden_size)
reps = mockingjay.forward(spec=spec, all_layers=False, tile=True)

# reps.shape: (batch_size, seq_len // downsample_rate, hidden_size)
reps = mockingjay.forward(spec=spec, all_layers=False, tile=False)
```
`spec` is the input spectrogram of the mockingjay model where:
- `spec` needs to be a PyTorch tensor with shape of `(seq_len, mel_dim)` or `(batch_size, seq_len, mel_dim)`.
- `mel_dim` is the spectrogram feature dimension which by default is `mel_dim == 160`, see [utility/audio.py](utility/audio.py) for more preprocessing details.

`reps` is a PyTorch tensor of various possible shapes where:
- `batch_size` is the inference batch size.
- `num_hiddem_layers` is the transformer encoder depth of the mockingjay model.
- `seq_len` is the maximum sequence length in the batch.
- `downsample_rate` is the dimensionality of the transformer encoder layers.
- `hidden_size` is the number of stacked consecutive features vectors to reduce the length of input sequences.

The output shape of `reps` is determined by the two arguments:
- `all_layers` is a boolean which controls whether to output all the Encoder layers, if `False` returns the hidden of the last Encoder layer.
- `tile` is a boolean which controls whether to tile representations to match the input `seq_len` of `spec`.

As you can see, `reps` is essentially the Transformer Encoder hidden representations in the mockingjay model. You can think of Mockingjay as a speech version of [BERT](https://arxiv.org/abs/1810.04805) if you are familiar with it.

There are many ways to incorporate `reps` into your downtream task. One of the easiest way is to take only the outputs of the last Encoder layer (i.e., `all_layers=False`) as the input features to your downstream model, feel free to explore other mechanisms.

## Fine-tuning with your own downstream SLP tasks
With this repo and the trained models, you can fine-tune the pre-trained Mockingjay model on your own dataset and tasks. To do so, take a look at the following example python code ([example_finetune.py](example_finetune.py)):
```python
import torch
from runner_mockingjay import get_mockingjay_model
from downstream.model import example_classifier
from downstream.solver import get_mockingjay_optimizer

# setup the mockingjay model
example_path = 'result/result_mockingjay/mockingjay_libri_sd1337_MelBase/mockingjay-500000.ckpt'
solver = get_mockingjay_model(from_path=example_path)

# setup your downstream class model
# features extracted from MelBase model have dimention 768
classifier = example_classifier(input_dim=768, hidden_dim=128, class_num=2).cuda()

# construct the Mockingjay optimizer
params = list(solver.mockingjay.named_parameters()) + list(classifier.named_parameters())
optimizer = get_mockingjay_optimizer(params=params, lr=4e-3, warmup_proportion=0.7, training_steps=50000)

# forward
example_inputs = torch.zeros(3, 800, 160) # A batch of spectrograms: (batch_size, seq_len, hidden_size)
reps = solver.forward_fine_tune(spec=example_inputs) # returns: (batch_size, seq_len, hidden_size)
loss = classifier(reps, torch.LongTensor([0, 1, 0]).cuda())

# update
loss.backward()
optimizer.step()

# save
PATH_TO_SAVE_YOUR_MODEL = 'example.ckpt'
states = {'Classifier': classifier.state_dict(), 'Mockingjay': solver.mockingjay.state_dict()}
torch.save(states, PATH_TO_SAVE_YOUR_MODEL)
```

# Requirements

- Python 3
- Pytorch 1.3.0 or above
- Computing power (high-end GPU) and memory space (both RAM/GPU's RAM) is **extremely important** if you'd like to train your own model.
- Required packages and their use are listed below, and also in [requirements.txt](requirements.txt):
```
editdistance     # error rate calculation
joblib           # parallel feature extraction & decoding
librosa          # feature extraction (for feature extraction only)
pydub            # audio segmentation (for MOSEI dataset preprocessing only)
pandas           # data management
tensorboardX     # logger & monitor
torch            # model & learning
tqdm             # verbosity
yaml             # config parser
matplotlib       # visualization
ipdb             # optional debugger
numpy            # array computation
scipy            # for feature extraction
```
The above packages can be installed by the command:
```bash
pip3 install -r requirements.txt
```
Below we list packages that need special attention, and we recommand you to install them manually:
```
apex             # non-essential, faster optimization (only needed if enabled in config)
sentencepiece    # sub-word unit encoding (for feature extraction only, see https://github.com/google/sentencepiece#build-and-install-sentencepiece for install instruction)
```

# Instructions

***Before you start, make sure all the packages required listed above are installed correctly***

### Step 0. Preprocessing - Acoustic Feature Extraction & Text Encoding

See the instructions on the [Preprocess wiki page](https://github.com/andi611/Mockingjay-Speech-Representation/wiki/Mockingjay-Preprocessing-Instructions) for preprocessing instructions.

### Step 1. Configuring - Model Design & Hyperparameter Setup

All the parameters related to training/decoding will be stored in a yaml file. Hyperparameter tuning and massive experiment and can be managed easily this way. See [config files](config/) for the exact format and examples.

### Step 2. Training the Mockingjay Model for Speech Representation Learning

Once the config file is ready, run the following command to train unsupervised end-to-end Mockingjay:
```bash
python3 runner_mockingjay.py --train
```
All settings will be parsed from the config file automatically to start training, the log file can be accessed through TensorBoard.

### Step 3. Using Pre-trained Models on Downstream Tasks

Once a Mockingjay model was trained, we can use the generated representations on downstream tasks.
See the [Experiment section](#Experiments) for reproducing downstream task results mentioned in our paper, and see the [Highlight section](#Highlight) for incorporating the extracted representations with your own downstream task.

Pre-trained models and their configs can be download from [HERE](http://bit.ly/result_mockingjay).
To load with default path, models should be placed under the directory path: `--ckpdir=./result_mockingjay/` and name the model file manually with `--ckpt=`.

### Step 4. Loading Pre-trained Models and Visualize
Run the following command to visualize the model generated samples:
```bash
# visualize hidden representations
python3 runner_mockingjay.py --plot
# visualize spectrogram
python3 runner_mockingjay.py --plot --with_head
```
Note that the arguments ```--ckpdir=XXX --ckpt=XXX``` needs to be set correctly for the above command to run properly.

### Step 5. Monitor Training Log
```bash
# open TensorBoard to see log
tensorboard --logdir=log/log_mockingjay/mockingjay_libri_sd1337/
# or
python3 -m tensorboard.main --logdir=log/log_mockingjay/mockingjay_libri_sd1337/
```

## Experiments

### Application on downstream tasks
See the instructions on the [Downstream wiki page](https://github.com/andi611/Mockingjay-Speech-Representation/wiki/Downstream-Task-Instructions) to reproduce our experiments.

### Comparing with APC
See the instructions on the [APC wiki page](https://github.com/andi611/Mockingjay-Speech-Representation/wiki/Reproducing-APC-to-compare-with-Mockingjay) to reproduce our experiments.


# Reference
1. [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/), McAuliffe et. al.
2. [CMU MultimodalSDK](https://github.com/A2Zadeh/CMU-MultimodalSDK/blob/master/README.md), Amir Zadeh.
3. [PyTorch Transformers](https://github.com/huggingface/pytorch-transformers), Hugging Face.
4. [Autoregressive Predictive Coding](https://arxiv.org/abs/1904.03240), Yu-An Chung.
5. [End-to-end ASR Pytorch](https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch), Alexander-H-Liu.
6. [Tacotron Preprocessing](https://github.com/r9y9/tacotron_pytorch), Ryuichi Yamamoto (r9y9)

## Citation
```
@misc{liu2019mockingjay,
    title={Mockingjay: Unsupervised Speech Representation Learning with Deep Bidirectional Transformer Encoders},
    author={Andy T. Liu and Shu-wen Yang and Po-Han Chi and Po-chun Hsu and Hung-yi Lee},
    year={2019},
    eprint={1910.12638},
    archivePrefix={arXiv},
    primaryClass={eess.AS}
}
```
