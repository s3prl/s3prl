# 👾 Mockingjay
## Speech Representation Learning through Self-Imitation - PyTorch Official Implementation

This is an open source project for Mockingjay, end-to-end learning of acoustic features representations, implemented with Pytorch.

Feel free to use/modify them, any bug report or improvement suggestion will be appreciated. If you have any questions, please contact r07942089[AT]ntu.edu.tw. If you find this project helpful for your research, please do consider to cite [this project](#Citation), thanks!

# Highlights

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

See the mockingjay [wiki page](https://github.com/andi611/Mockingjay-Speech-Representation-Learning/wiki/Mockingjay-Preprocessing) for preprocessing instructions.

## Step 1. Configuring - Model Design & Hyperparameter Setup

All the parameters related to training/decoding will be stored in a yaml file. Hyperparameter tuning and massive experiment and can be managed easily this way. See [documentation and examples](config/) for the exact format. **Note that the example configs provided were not fine-tuned**, you may want to write your own config for best performance.

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

## Step 5. Training and Testing of the Phone Classification Task
Run the following command to train a phone classifier:
```bash
# using spectrogram as baseline
python3 runner_mockingjay.py --train_phone
# using mockingjay representations
python3 runner_mockingjay.py --train_phone --run_mockingjay
```
Run the following command to test representations using the phone classifier:
```bash
# testing spectrogram as baseline
python3 runner_mockingjay.py --test_phone
# testing mockingjay representations
python3 runner_mockingjay.py --test_phone --run_mockingjay
```
Note that the arguments ```--ckpdir=XXX --ckpt=XXX``` needs to be set correctly for the above command to run properly.

## Step 6. Training and Testing of the Sentimental Classification Task
Run the following command to train a sentiment classifier:
```bash
# using spectrogram as baseline
python3 runner_mockingjay.py --train_sentiment
# using mockingjay representations
python3 runner_mockingjay.py --train_sentiment --run_mockingjay
```
Run the following command to test representations using the phone classifier:
```bash
# testing spectrogram as baseline
python3 runner_mockingjay.py --test_sentiment
# testing mockingjay representations
python3 runner_mockingjay.py --test_sentiment --run_mockingjay
```
Note that the arguments ```--ckpdir=XXX --ckpt=XXX``` needs to be set correctly for the above command to run properly.

## Step 7. Training and Testing of the Speaker Verification Task
Run the following command to train a sentiment classifier:
```bash
# using spectrogram as baseline
python3 runner_mockingjay.py --train_speaker
# using mockingjay representations
python3 runner_mockingjay.py --train_speaker --run_mockingjay
```
Run the following command to test representations using the phone classifier:
```bash
# testing spectrogram as baseline
python3 runner_mockingjay.py --test_sentiment
# testing mockingjay representations
python3 runner_mockingjay.py --test_sentiment --run_mockingjay
```
Note that the arguments ```--ckpdir=XXX --ckpt=XXX``` needs to be set correctly for the above command to run properly.

## Step 8. Monitor Training Log
```bash
# open TensorBoard to see log
tensorboard --logdir=log_mockingjay/mockingjay_libri_sd1337/
# or
python3 -m tensorboard.main --logdir=log_mockingjay/mockingjay_libri_sd1337/
```


# Reference
1. [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/), McAuliffe et. al.
2. [CMU-MultimodalSDK](https://github.com/A2Zadeh/CMU-MultimodalSDK/blob/master/README.md), Amir Zadeh.
3. [PyTorch-Transformers](https://github.com/huggingface/pytorch-transformers), Hugging Face.
4. [End-to-end-ASR-Pytorch](https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch), Alexander-H-Liu.


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
