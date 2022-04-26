# MOSEI

This directory already contains `CMU_MOSEI_Labels.csv` to train 
sentiment analysis and emotion predictions from CMU-MOSEI dataset. 
However, you can reproduce the generated `CMU_MOSEI_Labels.csv` by yourself 
by following steps in `utility` directory.


## Data preparation

### Download Dataset

```bash=
cd /path/to/data

# Download CMU-MOSEI Dataset
wget http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSEI.zip
unzip CMU_MOSEI.zip 

# Set final path to dataset
mv Raw CMU-MOSEI
```
Remember to modify data_dir in `config.yaml`

### Install Dependencies

Install dependencies using the following commands:
```bash=
pip install pydub pandas
sudo apt-get install ffmpeg libavcodec-extra
```

### Segment the Audio Files

Run `utility/segment_audio.py` by passing the location of your CMU-MOSEI Audio folder as an argument.
This script is being used to segment the audio files into different split (train, dev, and test).

```bash=
python ./utility/segment_audio.py /path/to/CMU_MOSEI/Audio
# For example: python ./utility/segment_audio.py /tmp/CMU_MOSEI/Audio
```

After that, you should have the following file structure under /path/to/CMU_MOSEI/Audio:

```
/path/to/CMU_MOSEI/Audio
├── Full
│   ├── COVAREP
│   │   └── Many .mat files
│   └── WAV_16000
│       └── Many .wav files
└── Segmented_Audio
    ├── dev
    │   └── Many .wav files (Dev set)
    ├── test
    │   └── Many .wav files (Test set)
    └── train
        └── Many .wav files (Train set)
```

## Available Tasks
Change `num_class` in the `config.yaml` to perform the following tasks:   
1. **Two-class sentiment classication (`num_class: 2`)**  
Labels: [0: negative, 1: non-negative] (positive and neutral are counted as non-negative).  
2. **Three-class sentiment classification (`num_class: 3`)**  
Labels: [-1: negative, 0: neutral, 1: positive]  
3. **Six-class emotion classification (`num_class: 6`)**  
Labels: [happy, sad, anger, surprise, disgust, fear]  
4. **Seven-class sentiment classification (`num_clases: 7`)**  
Labels: [-3: highly negative, -2: negative, -1: weakly negative, 0: neutral, 1: weakly positive, 2: positive, 3 highly positive]

## Customize Downstream Model
If you want to modify the model architecture used for downstream training, please modify the content of `model.py` and `config.yaml`. For example, if you want to use a transformer-encoder-based model:
- In `model.py`, change Model to:
```python=
class Model(nn.Module):
    def __init__(self, input_dim, output_class_num, **kwargs):
        super(Model, self).__init__()
        self.enc_layer = nn.TransformerEncoderLayer(
            input_dim, **kwargs['enc']
        )
        self.encoder = nn.TransformerEncoder(
            self.enc_layer, kwargs['enc_layers']
        )
        self.linear = nn.Sequential(
            nn.Linear(input_dim, output_class_num)
        )

    def forward(self, features):
        x = self.encoder(features)
        pooled = x.mean(dim=1)
        predicted = self.linear(pooled)
        return predicted
```

- In `config.yaml`, change modelrc to:
```python=
  modelrc:
    input_dim: 256
    enc:
      nhead: 2
      batch_first: True
    enc_layers: 2
```

## Train a New Model

Use the following code to train a model using CMU-MOSEI as downstream dataset:
```
# Training w/o DDP
python run_downstream.py -m train -n ExpName -u fbank -d mosei

# Training w/ DDP
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 run_downstream.py -m train -n ExpName -u fbank -d mosei

# Testing
python run_downstream.py -m evaluate -n ExpName -u fbank -d mosei
```
Consequently, you can calculate the accuracy and other metrics manually from the generated txt files.

