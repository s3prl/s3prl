# Mockyingjay - PyTorch Implementation

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
python3 preprocess.py 
```
Run preprocessing with the following command to change input directory:
```
python3 preprocess.py --data_path <path to LibriSpeech on your computer> 
```
You may check the parameter type and default value by using the option ```--help``` for each script.

### Step 1. Configuring - Model Design & Hyperparameter Setup

All the parameters related to training/decoding will be stored in a yaml file. Hyperparameter tuning and massive experiment and can be managed easily this way. See [documentation and examples](config/) for the exact format. **Note that the example configs provided were not fine-tuned**, you may want to write your own config for best performance.

### Step 2. Training End-to-end ASR (or RNN-LM) Learning

Once the config file is ready, run the following command to train end-to-end ASR (or language model)
```
python3 runner_asr.py --config <path of config file> 
```
All settings will be parsed from the config file automatically to start training, the log file can be accessed through TensorBoard. ***Please notice that the error rate reported on the TensorBoard is biased (see issue #10), you should run the testing phase in order to get the true performance of model***. For example, train an ASR on LibriSpeech and watch the log with
```
python3 runner_asr.py --config config/asr_libri.yaml
# open TensorBoard to see log
tensorboard --logdir log/
# Train an external language model
python3 runner_asr.py --config config/rnnlm_libri.yaml --rnnlm
```

### Step 3. Testing - Speech Recognition & Performance Evaluation

Once a model was trained, run the following command to test it
```
python3 runner_asr.py --config <path of config file> --test
```
Recognition result will be stored at `result/<name>/` as a txt file with auto-naming according to the decoding parameters specified in config file. The result file may be evaluated with `eval.py`. For example, test the ASR trained on LibriSpeech and check performance with
```
python3 runner_asr.py --config config/asr_libri.yaml --test
# Check WER/CER
python3 runner_asr.py --file result/libri_example_sd0/decode_*.txt
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
  title={},
  author={Liu, Andy T. and Lee, Hung-yi},
  booktitle={},
  year={2019},
  organization={}
}
```
