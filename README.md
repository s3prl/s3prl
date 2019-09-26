# 👾 Mockingjay
## Speech Representation Learning with Deep Bidirectional Transformer Encoders
### PyTorch Official Implementation

This is an open source project for Mockingjay, end-to-end learning of acoustic features representations, implemented with Pytorch.

Feel free to use/modify them, any bug report or improvement suggestion will be appreciated. If you have any questions, please contact r07942089[AT]ntu.edu.tw. If you find this project helpful for your research, please do consider to cite [this project](#Citation), thanks!

# Highlights
With this repo and trained models, you can use it to extract speech representations from your target dataset. To do so, feed-forward the trained model on the target dataset and retrieve the extracted features by running the following example python code:
```python
from runner_mockingjay import get_mockingjay_model
path = 'result/result_mockingjay/mockingjay_libri_sd1337_best/mockingjay-500000.ckpt'
mockingjay = get_mockingjay_model(from_path=path)

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
- `mel_dim` is the spectrogram feature dimension which by default is `mel_dim == 160`, see [utils/audio.py](utils/audio.py) for more preprocessing details.

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

# Requirements

- Python 3
- Computing power (high-end GPU) and memory space (both RAM/GPU's RAM) is **extremely important** if you'd like to train your own model.
- Required packages and their use are listed:
```
apex             # non-essential, faster optimization (only needed if enabled in config)
editdistance     # error rate calculation
joblib           # parallel feature extraction & decoding
librosa          # feature extraction (for feature extraction only)
pandas           # data management
sentencepiece    # sub-word unit encoding (for feature extraction only, see https://github.com/google/sentencepiece#build-and-install-sentencepiece for install instruction)
tensorboardX     # logger & monitor
torch            # model & learning
tqdm             # verbosity
yaml             # config parser
mmsdk            # sentiment dataset CMU-MOSI SDK (sentiment data preprocessing only, see https://github.com/A2Zadeh/CMU-MultimodalSDK#installation for install instruction)
```

# Instructions

***Before you start, make sure all the packages required listed above are installed correctly***

## Step 0. Preprocessing - Acoustic Feature Extraction & Text Encoding

See the instructions on the [Preprocess wiki page](https://github.com/andi611/Mockingjay-Speech-Representation-Learning/wiki/Mockingjay-Preprocessing-Instructions) for preprocessing instructions.

## Step 1. Configuring - Model Design & Hyperparameter Setup

All the parameters related to training/decoding will be stored in a yaml file. Hyperparameter tuning and massive experiment and can be managed easily this way. See [config files](config/) for the exact format and examples.

## Step 2. Training the Mockingjay Model for Speech Representation Learning

Once the config file is ready, run the following command to train unsupervised end-to-end Mockingjay:
```bash
python3 runner_mockingjay.py --train
```
All settings will be parsed from the config file automatically to start training, the log file can be accessed through TensorBoard.

## Step 3. Loading Pre-trained Models and Testing

Once a model was trained, run the following command to test the generated representations:
```bash
python3 runner_mockingjay.py --load --test_phone
```
Pre-trained models and their configs can be download from [HERE](https://drive.google.com/drive/folders/1tZQnT8y7sE6kuxVWivo-KmRw8CgLy7da?usp=sharing).
To load with default path, models should be placed under the directory path: `--ckpdir=./result_mockingjay/` and name the model file manually with `--ckpt=`.

## Step 4. Loading Pre-trained Models and Visualize
Run the following command to visualize the model generated samples:
```bash
# visualize spectrogram
python3 runner_mockingjay.py --plot
# visualize hidden representations
python3 runner_mockingjay.py --plot --with-head
```
Note that the arguments ```--ckpdir=XXX --ckpt=XXX``` needs to be set correctly for the above command to run properly.

## Step 5. Monitor Training Log
```bash
# open TensorBoard to see log
tensorboard --logdir=log/log_mockingjay/mockingjay_libri_sd1337/
# or
python3 -m tensorboard.main --logdir=log/log_mockingjay/mockingjay_libri_sd1337/
```

## Experiments - Application on downstream tasks
See the instructions on the [Downstream wiki page](https://github.com/andi611/Mockingjay-Speech-Representation-Learning/wiki/Downstream-Task-Instructions) to reproduce our experiments.

## Experiments - Compare with APC
See the instructions on the [APC wiki page](https://github.com/andi611/Mockingjay-Speech-Representation-Learning/wiki/Reproducing-APC-to-compare-with-Mockingjay) to reproduce our experiments.


# Reference
1. [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/), McAuliffe et. al.
2. [CMU MultimodalSDK](https://github.com/A2Zadeh/CMU-MultimodalSDK/blob/master/README.md), Amir Zadeh.
3. [PyTorch Transformers](https://github.com/huggingface/pytorch-transformers), Hugging Face.
4. [Autoregressive Predictive Coding](https://arxiv.org/abs/1904.03240), Yu-An Chung.
5. [End-to-end ASR Pytorch](https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch), Alexander-H-Liu.


## Citation
```
@inproceedings{,
  title={Mockingjay: Speech Representation Learning through Self-Imitation},
  author={Liu, Andy T. and Lee, Hung-yi},
  booktitle={},
  year={2019},
  organization={College of Electrical Engineering and Computer Science, National Taiwan University}
}
```
