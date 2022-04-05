# MOSEI

This directory already contains `CMU_MOSEI_Labels.csv` to train 
sentiment analysis and emotion predictions from CMU-MOSEI dataset. 
However, you can reproduce the generated `CMU_MOSEI_Labels.csv` by yourself 
by following steps in `utility` directory.


## Data preparation

### Download the Dataset

```bash=
cd /path/to/data

# Download CMU-MOSEI Dataset
wget http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSEI.zip
unzip CMU_MOSEI.zip 

# Set final path to dataset
mv Raw CMU_MOSEI
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


## Train a New Model

If you want to modify the model architecture used for downstream training, please refer to `model.py` and modify the content.

Use the following code to train a model using CMU-MOSEI as downstream dataset:
```
# Training w/o DDP
python run_downstream.py -m train -n ExpName -u fbank -d mosei

# Training w/ DDP
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node 2 run_downstream.py -m train -n ExpName -u fbank -d mosei

# Testing
python run_downstream.py -m evaluate -n ExpName -u fbank -d mosei
```
Consequently, you can calculate the accuracy and other metrics manually from the generated txt files.

