# Mockingjay: Speech Representation Learning through Self-Imitation - PyTorch Official Implementation

This is an open source project for Mockingjay, end-to-end learning of acoustic features representations, implemented with Pytorch.

Feel free to use/modify them, any bug report or improvement suggestion will be appreciated. If you have any questions, please contact r07942089[AT]ntu.edu.tw. If you find this project helpful for your research, please do consider to cite [this project](#Citation), thanks!

## Highlights

## Requirements

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
```

## Instructions

***Before you start, make sure all the packages required listed above are installed correctly***

### Step 0. Preprocessing - Acoustic Feature Extraction & Text Encoding

Preprocessing scripts may be executed directly if the LibriSpeech dataset is placed under [`data/`](data/). The extracted data, which is ready for training, will be stored under the same [`data/`](data/) directory by default. 
```
# Defualt
python3 preprocess.py --feature_type=fbank
# To train on different input / output target features:
python3 preprocess.py --feature_type=mel
python3 preprocess.py --feature_type=linear 
```
Run preprocessing with the following command to change input directory:
```
python3 preprocess.py --data_path <path to LibriSpeech on your computer> 
```
You may check the parameter type and default value by using the option ```--help``` for each script.

### Step 1. Configuring - Model Design & Hyperparameter Setup

All the parameters related to training/decoding will be stored in a yaml file. Hyperparameter tuning and massive experiment and can be managed easily this way. See [documentation and examples](config/) for the exact format. **Note that the example configs provided were not fine-tuned**, you may want to write your own config for best performance.

### Step 2. Training the Mockingjay Model for Speech Representation Learning

Once the config file is ready, run the following command to train unsupervised end-to-end Mockingjay:
```
python3 runner_mockingjay.py
```
All settings will be parsed from the config file automatically to start training, the log file can be accessed through TensorBoard.

### Step 3. Loading Pre-trained Models and Testing

Once a model was trained, run the following command to get the generated representations:
```
python3 runner_mockingjay.py --test --load --ckpdir='directory_to_model' --ckpt='model_name'
```
Run the following command to visualize the model generated samples:
```
# spectrogram
python3 runner_mockingjay.py --test --load --plot
# hidden representations
python3 runner_mockingjay.py --test --load --plot --with-head
```
Pre-trained models and their configs can be download from [HERE](https://drive.google.com/drive/folders/1tZQnT8y7sE6kuxVWivo-KmRw8CgLy7da?usp=sharing).
To load with default path, models should be placed under this directory path: `--ckpdir=./result_mockingjay/` and name the model file with `--ckpt=`.

### Step 4. Monitor Training Log
```
# open TensorBoard to see log
tensorboard --logdir=log_mockingjay/mockingjay_libri_sd1337/
# or
python3 -m tensorboard.main --logdir=log_mockingjay/mockingjay_libri_sd1337/
```

## ToDo
- 

## Acknowledgements 
- ASR Implementation by [Alexander-H-Liu](https://github.com/Alexander-H-Liu), a great end-to-end Automatic Speech Recognition System.


## Reference
1. [End-to-end-ASR-Pytorch](https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch), Alexander-H-Liu.


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
