# Tasks used in Mockingjay & TERA

## Prerequisite

Please read [downstream/README.md](../README.md) for the general command pattern, and read [upstream/README.md](../../upstream/README.md) for registering a new pretrained model (upstream).

## Introduction

This document includes the tasks used in [Mockingjay](https://arxiv.org/abs/1910.12638) & [TERA](https://arxiv.org/abs/2007.06028). If you use them for your research please consider citing these papers!

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

## Phone Classification - LibriSpeech

Specified by the command `-d` (with different variants):
- `phone_linear`
- `phone_linear_concat` 
- `phone_1hidden`

#### Prepare data
1. Download the raw [LibriSpeech](https://www.openslr.org/12) corpus and unzip.

    ```bash
    cd /path/to/put/data
    wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
    tar zxvf train-clean-100.tar.gz
    ```

2. After extracting the file, you should have the file structure as following:

    ```bash
    LibriSpeech
    ├── train-clean-100
    └── README.TXT
    ```

3.  *(Optional)* Allow bucketing to increase training efficientcy & speed, this will generate a directory called `data/len_for_bucket`:

    ```bash
    python preprocess/generate_len_for_bucket.py --data_root "your_libri_root" --output_path ./data/
    ```

4. Change the following paths under `phone_*/config.yaml` to your own:

    ```yaml
    libri_root: '/media/andi611/1TBSSD/LibriSpeech/'
    bucket_file: 'data/len_for_bucket'
    ```

#### Training

```bash
python run_downstream.py -m train -u baseline -d phone_linear -n ExpName
python run_downstream.py -m train -u baseline -d phone_linear_concat -n ExpName
python run_downstream.py -m train -u baseline -d phone_1hidden -n ExpName
```

#### Testing

Testing is done on-the-fly with training since it is not costly. Use the following command to get the testing result from the best-dev ckpt:

```bash
python utility/get_best_dev.py result/downstream/ExpName/log.log
```

## Speaker Classification - LibriSpeech

TODO

## Sentiment analysis - CMU-MOSEI

- Prepare data:
    1) Download and unzip data to the path you want:
    ```bash
    cd /path/to/put/data
    wget http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSEI.zip
    unzip CMU_MOSEI.zip
    ```
    2) After extracting the file, you should have the file structure as following. You only need to keep the folder "Audio"
    ```bash
    CMU_MOSEI
    ├── Audio
    │   └── FULL
    ├── Videos
    │   └── ..
    ├── ..
    ```
    
    3) Change the following paths under `mosei/segment_audio.sh` to your own. 
    ```sh
    python3 ./utility/segment_audio.py **/home/godiclili/Audio**
    ```
    
    4) Segment Audio by running `mosei/segment_audio.sh`
    ```bash
    bash segment_audio.sh
    ```
    
    5) Change the following paths under `mosei/config.yaml` to your own.
    ```yaml
    data_dir: **/home/godiclili/Audio**
    ```
    
    6) Specify number of classes (2/7, default 2) for classification under `mosei/config.yaml`
    ```yaml
    num_class: 2
    ```
- Example run command (with a pseudo upstream):
```
python3 run_downstream.py -m train -u baseline -d mosei -n HelloWorld
```
