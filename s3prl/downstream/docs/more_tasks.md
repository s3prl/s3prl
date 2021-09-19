# More tasks

## Prerequisite

Please read [downstream/README.md](../README.md) for the general command pattern, and read [upstream/README.md](../../upstream/README.md) for registering a new pretrained model (upstream).

## Introduction

This document includes a lot more tasks to try! However, they might be out-of-date and need a little update to match the latest coding style and be runnable. Any kind of contribution is welcomed!

## Trainable Spoken Term Detection - SWS2013

Specified by the command `-d sws2013`

#### Prepare data
1. Download the SWS2013
    - https://speech.fit.vutbr.cz/files/sws2013Database.tgz

2. Specify the place to unpack the database

    ```bash
    export CORPORA_DIR=/YOUR/CORPORA/DIR/PATH
    ```

3. Unpack the tarball

    ```bash
    tar zxf sws2013Database.tgz -C $CORPORA_DIR
    ```

4. Further unpack the scoring script tarball

    ```bash
    tar zxf $CORPORA_DIR/sws2013Database_dev_eval/scoring_atwv_sws2013_full.tgz -C $CORPORA_DIR/sws2013Database_dev_eval
    ```

5. Change the following path in `sws2013/config.yaml` to yours
    ```yaml
    sws2013_root: /YOUR/CORPORA/DIR/PATH/sws2013Database_dev_eval
    sws2013_scoring_root: /YOUR/CORPORA/DIR/PATH/sws2013Database_dev_eval/scoring_atwv_sws201
    ```

#### Train
```bash
python3 run_downstream.py -m train -u fbank -d sws2013 -n ExpName
```

## Intent Classification - SNIPS
- Variants to this task:
    None
- Prepare data:
    1) Prepare the Audio file:
    ```bash
    cd /path/to/put/data
    wget https://shangwel-asr-evaluation.s3-us-west-2.amazonaws.com/audio_slu_v3.zip
    unzip audio_slu_v3.zip
    ```
    2) Prepare the NLU annotation file:
    ```bash
    git clone https://github.com/aws-samples/aws-lex-noisy-spoken-language-understanding.git
    cp -r aws-lex-noisy-spoken-language-understanding/* audio_slu
    ```
    3) After extracting the file, you should have the file structure as following: 
    ```bash
    audio_slu
    ├── data
    │   └── nlu_annotation
    │       └── [*.csv]
    ├── license
    ├── audio_Aditi
    ...
    └── audio_Salli
    ```
    4) Change the following paths under `audio_snips/config.yaml` to your own and specify speakers you want in training set and test set:
    ```yaml
    file_path: /home/raytz/Disk/data/audio_slu
    train_speakers: 
      - Aditi
      ...
      - Salli
    test_speakers:
      - Aditi
      ...
      - Salli
    ```
- Example run command (with a pseudo upstream):
```
python3 run_downstream.py -m train -u baseline -d audio_snips -n HelloWorld
```

## Intent Classification - ATIS
- Variants to this task:
    None
- Prepare data:
    1) Prepare the dataset (under the folder of /groups/public):
    ```bash
    //first sftp to the battleship
    lcd /path/to/put/data
    get -r /groups/public/atis
    ```
    2) After downloading the dataset, you should have the file structure as following: 
    ```bash
    atis
    ├── test
    ├── nlu_iob
    ├── train
    ├── dev
    ├── all.trans.txt
    ├── all.iob.trans.txt
    └── slot_vocabs.txt
    ```
    4) Change the following paths under `audio_snips/config.yaml` to your own:
    ```yaml
    file_path: /home/raytz/Disk/data/atis
    ```
- Example run command (with a pseudo upstream):
```
python3 run_downstream.py -m train -u baseline -d atis -n HelloWorld
```
