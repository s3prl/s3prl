# Downstream tasks


### Introduction
Self-supervised (SSL) pretrained models are not able to justify their effectiveness through pretraining loss. One always has to evaluate their performance with downstream tasks. Hence, it is crucial to collect a wide range of downstream tasks and make the evaluation pipeline as easy as possible to speed up the development cycle.

We develop several downstream tasks for evaluating SSL models, each of them is defined by a sub-folder under this **downstream** folder. We further select representative ones to form the following **benchmark**:
- [**SUPERB:** **S**peech processing **U**niversal **PER**formance **B**enchmark](https://arxiv.org/abs/2105.01051)

### How to use

#### I. General requirement

1. [**Clone the repository and install dependencies**](../README.md#installation)
2. See the [**General Usage**](#general-usage) to have a sense on the conceptual usage

#### II A. Run the developed tasks

3. **Optional:** [**Register your customized pretrained model** (will be detailed soon)]()
    - You can also start from evaluating [pretrained models available in this toolkit](../upstream/README.md#upstream-self-supervised-models).
4. Follow the task-specific usages
    - [**SUPERB Benchmark**](#superb-benchmark)
    - [More tasks](#more-tasks)

#### II B. Develop new tasks

3. Check [**Add new downstream tasks**](#add-new-downstream-tasks). Pull requests are always welcome. Thanks!


# General usage
All of the downstream task follow the following command pattern, with a few task-specific adjustments which are detailed in the follow-up task-specific sections.

## Start a new downstream training experiment

```bash
# general pattern
python3 run_downstream.py -m train -n ExpName -u UpstreamName -d DownstreamName
# a directly runnable example without data preparation
python3 run_downstream.py -m train -n ExpName -u fbank -d example
```

- `-m` or `--mode` specifies the **train/evaluate** mode
- `-u` or `--upstream` specifies the upstream pretrained model.
    - The available upstream can be checked by `-h`
- `-d` or `--downstream` specifies the downstream task.
    - The available downstream can be checked by `-h`
    - Each available downstream task has its corresponding folder under `downstream/`. Eg. `-d asr` means we are using the task defined in `downstream/asr/`
    - `example` is a pseudo downstream task which is useful for testing the upstream model or as an initial template for developing a new downstream task
- `-f` or `--upstream_trainable` enables finetuning the upstream model on the downstream task. Default: false 
- `-n` or `--name` specifies the experiment name, all the files related to this run will be saved into **expdir**=`result/downstream/{args.name}`. (You can also use `-p` or `--expdir` to directly specify the path of **expdir**.)
    - command
    - config file
    - Tensorboard event file
    - checkpoints, each contains
        - arguments
        - config
        - latest optimization step
        - latest optimization epoch
        - **state_dict** of models, optimizer, scheduler
- `-c` or `--config` specifies the config file path. If not specified, will use the `config.yaml` under each downstream folder by default. Eg. `result/asr/config.yaml`
- `-o` or `--override` can override any argument or config field with command line, which is at the highest priority. Please refer to the [override function](../utility/helper.py) for definition. Here is an example to override 3 fields defined in this [config file](./example/config.yaml):

    ```bash
    -o "config.optimizer.lr=1.0e-3,,config.optimizer.name='AdamW',,config.runner.eval_dataloaders=['dev', 'test']"
    ```

## Resume training from a checkpoint
```bash
# [ckpt] can be the path of a checkpoint or its residing directory.
python3 run_downstream.py -m train -e [ckpt]
```

- The `-e` or `--past_exp` option is designed to use the exact same arguments and config as the previous training experiment **except the training/evaluation mode**. (Each checkpoint will save arguments and config.)
- `-o` can be used to further override the arguments & configs loaded from the checkpoint, since `-o` is at the highest priority.


## Fault-tolerant training

```bash
for i in $(seq 1 100); do
    python3 run_downstream.py -m train -n ExpName -u fbank -d example -a
done
```

- The `-a` option stands for **automatic resuming**, will resume the checkpoint when there is a latest checkpoint resides in `expdir` directory or start a new training experiment when there is none.

`run_while.sh` under the root directory of the repo is a helping wrapper for this. For any **COMMAND** you wish to run in a while loop, you can just
```bash
./run_while.sh COMMAND
```
Eg.
```bash
./run_while.sh python3 run_downstream.py -a -m train -n ExpName -u fbank -d example
```
Please must remember to use `-a` when wrap with `run_while.sh`, or else you are going to re-launch a new training experiment for every loop, which will be a disaster expecially for Tensorboard event files.

## Distributed training

We wrap the model with **DistributedDataParallel**. By inserting `-m torch.distributed.launch --nproc_per_node {GPU_NUM}` between `python3` and `run_downstream.py`, you can directly turn the above **training** commands into distributed training. Currently only [ASR](#asr-automatic-speech-recognition) and [ASV](#asv-automatic-speaker-verification) support distributed training.

#### First specify your GPU number
```bash
gpus=16;
distributed="-m torch.distributed.launch --nproc_per_node ${gpus}";
```

#### Simple training
```bash
python3 $distributed run_downstream.py -m train -n ExpName -u fbank -d example
```

#### Resume training

```bash
# The $distributed value should be same as the original training experiment.
# [ckpt] can be the path of a checkpoint or its residing directory.
python3 $distributed run_downstream.py -m train -e [ckpt]
```

#### Fault-tolerant training

```bash
for i in $(seq 1 100); do
    python3 $distributed run_downstream.py -m train -n ExpName -u fbank -d example -a
    # When one of the spawned process dies, sometimes not all processes are terminated synchronizely.
    # You might need to ensure all the spawned process are killed here.
    # `killall` linux command is suitable for this.
done
```

## Test a checkpoint

The following `test-clean` is an example for the **name** of the testing dataset, and the supported **name** is defined by each downstream expert's `get_dataloader`. Typically `dev` and `test` are supported for task/dataset with the standard split.

### Preferable: Use the same args & config as training time

```bash
# [ckpt] can be the path of a checkpoint or its residing directory.
python3 run_downstream.py -m evaluate -t "test-clean" -e [ckpt]
```

- The `-e` or `--past_exp` option is designed to use the exact same arguments and config as the previous training experiment **except the training/evaluation mode**. (Each checkpoint will save arguments and config.)
- `-o` can be used to further override the arguments & configs loaded from the checkpoint, since `-o` is at the highest priority.

### Alternative: Use another set of args & config for testing

Most of the time the above command is enough. But if you find overridding args & configs stored in the trained checkpoint one-by-one cumbersome, you can first prepare a new set of args & config and only load the model weights in the trained checkpoint.

```bash
# [ckpt] can be the path of a checkpoint or its residing directory.
# [upstream], [downstream] and other args should be taken care by the user and won't loaded from the checkpoint.
# [config] is the newly prepared testing config
python3 run_downstream.py -m evaluate -t "test-clean" -i [ckpt] -u [upstream] -d [downstream] -c [config] -n TestExpName
```

- The `-i` or`--init_ckpt` option is designed to load a checkpoint without overwriting args & config, which enables flexible configuration for testing stage while user should take care of using the same upstream & downstream arguments as training time. Since the command and configs will all be saved into **expdir**, you can double check the setting by files in **expdir** of the previous training experiment.


### Test the distributed trained checkpoint

Only the training part is powered by **DistributedDataParallel**, and we save all the model *state_dict* **without** the DDP wrapper. That is, after the DDP training, you can always evaluate the checkpoint using the testing command documented above (on single GPU).

# SUPERB Benchmark

In this section we detail the commands for reproducing the paper [**SUPERB:** **S**peech processing **U**niversal **PER**formance **B**enchmark](https://arxiv.org/abs/2105.01051).

## PR: Phoneme Recognition

Specified by the command `-d ctc`

#### Prepare data

1. Download [LibriSpeech](https://www.openslr.org/12) and unzip. Only need train-clean-100, dev-clean, and test-clean.

2. Check the prepared file structure

    ```bash
    LibriSpeech/
    ├── train-clean-100/
    ├── dev-clean/
    └── test-clean/
    ```

3. Change the path in `downstream/ctc/libriphone.yaml`

    ```yaml
    downstream_expert:
        corpus:
            path: "root directory of LibriSpeech"
    ```

#### Training

```bash
python3 run_downstream.py -n ExpName -m train -u fbank -d ctc -c downstream/ctc/libriphone.yaml
```

#### Testing

```bash
python3 run_downstream.py -m evaluate -e result/downstream/ExpName/dev-best.ckpt
```

## ASR: Automatic Speech Recognition

Specified by the command `-d asr`

#### Prepare data

1. Download [LibriSpeech](https://www.openslr.org/12) and unzip. Only need train-clean-100, dev-clean, and test-clean.

2. Check the prepared file structure

    ```bash
    LibriSpeech/
    ├── train-clean-100/
    ├── dev-clean/
    └── test-clean/
    ```

3. Change the path in `downstream/asr/config.yaml`

    ```yaml
    downstream_expert:
        datarc:
            libri_root: "root directory of LibriSpeech"
    ```

4. Prepare the lengths for utterances in LibriSpeech's train-clean-100, dev-clean and test-clean:

    ```bash
    # Official LibriSpeech is in .flac format
    python3 preprocess/generate_len_for_bucket.py -i "root directory of LibriSpeech" -o data/librispeech -a .flac --n_jobs 12
    ```

#### Training

```bash
python3 run_downstream.py -n ExpName -m train -u fbank -d asr
```

#### Testing without LM

```bash
python3 run_downstream.py -m evaluate -t "test-clean" -e result/downstream/dev-clean-best.ckpt
```

#### Testing with KenLM + LibriSpeech official 4-gram LM

##### I. Prepare Decoding Environment

1. Install [KenLM](https://github.com/kpu/kenlm)
    - Please follow the official installation instructions of KenLM instead of the one documented in flashlight or wav2letter
    - If you encounter issues when installing KenLM, you might need to install some [extra dependencies](https://medium.com/tekraze/install-kenlm-binaries-on-ubuntu-language-model-inference-tool-33507000f33).

2. Install [flashlight python bindings](https://github.com/flashlight/flashlight/blob/master/bindings/python/README.md)
    - Only the **python bindings** is required instead of the entire flashlight toolkit

3. Download LibriSpeech official 4-gram LM
    - https://www.openslr.org/resources/11/4-gram.arpa.gz
    - Downloaded filename: **4-gram.arpa.gz**

4. Download character-based lexicon
    - https://dl.fbaipublicfiles.com/fairseq/wav2vec/librispeech_lexicon.lst
    - Downloaded filename: **librispeech_lexicon.lst**

5. Make sure your fairseq version contains the following commit
    - https://github.com/pytorch/fairseq/commit/cb84694c195afced474d17318b5e746d1a9d20a3#diff-ee3a94b6d9b5f2cc60f1b69afc075abbe2061083b52515178eb7145d59e7e7e4

##### II. Test

```bash
python3 run_downstream.py -m evaluate -t "test-clean" -e result/downstream/dev-best.ckpt \
    -o "\
        config.downstream_expert.datarc.decoder_args.decoder_type='kenlm',, \
        config.downstream_expert.datarc.decoder_args.kenlm_model='/path/to/4-gram.arpa.gz',, \
        config.downstream_expert.datarc.decoder_args.lexicon='/path/to/librispeech_lexicon.lst' \
       "
```

## KS: Keyword Spotting

Specified by the command `-d speech_commands`

#### Prepare data

1. Download data
    - http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz
    - http://download.tensorflow.org/data/speech_commands_test_set_v0.01.tar.gz

2. Download and unpack Speech Commands

    ```bash
    mkdir -p /CORPORA_DIR/speech_commands_v0.01
    tar zxf speech_commands_v0.01.tar.gz -C /CORPORA_DIR/speech_commands_v0.01
    ```

3. Download and unpack Speech Commands test set

    ```bash
    mkdir -p /CORPORA_DIR/speech_commands_test_set_v0.01
    tar zxf speech_commands_test_set_v0.01.tar.gz -C /CORPORA_DIR/speech_commands_test_set_v0.01
    ```

4. Change the following path in `downstream/speech_commands/config.yaml` to yours

    ```yaml
    downstream_expert:
        datarc:
            speech_commands_root: "/CORPORA_DIR/speech_commands_v0.01/"
            speech_commands_test_root: "/CORPORA_DIR/speech_commands_test_set_v0.01/"
    ```

#### Training

```bash
python3 run_downstream.py -n ExpName -m train -u fbank -d speech_commands
```

#### Testing

The testing is done on-the-fly with training since it is not costly. Use the following command to get the testing result from the best-dev checkpoint

```bash
python3 utility/get_best_dev.py result/downstream/ExpName/log.log
```

#### Compatible with Speech Command v2

The implementation is directly compatible with Speech Command v2. You can enable this by just changing the train/test dataset. All other steps should be the same.

- http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
- http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz

## QbE: Query-by-Example Spoken Term Detection

Specified by the command `-d quesst14_dtw`. This task does not require training. We extract representations and run dynamic time warping (DTW) on them.

#### Prepare data

1. Download QUESST14

    ```bash
    export CORPORA_DIR="the root directory of all your datasets"    
    wget https://speech.fit.vutbr.cz/files/quesst14Database.tgz
    tar zxf quesst14Database.tgz -C $CORPORA_DIR
    ```

2. Change the path in `downstream/quesst14/config.yaml`
   ```yaml
   downstream:
       datarc:
           dataset_root: "CORPORA_DIR/quesst14Database"
   ```

#### Dynamic Time Warping (DTW)

```bash
# The default dist_fn if not specified is "cosine_exp"
# as it yields the best result for almost all upstream
# Supported dist_fn: cosine, cityblock, euclidean, cosine_exp

dist_fn=cosine;

# dev
python3 run_downstream.py -m evaluate -t "dev" -u fbank -d quesst14_dtw \
    -n ExpName_dev -o "config.downstream_expert.dtwrc.dist_method='$dist_fn'"

# test
python3 run_downstream.py -m evaluate -t "test" -u fbank -d quesst14_dtw \
    -n ExpName_test -o "config.downstream_expert.dtwrc.dist_method='$dist_fn'"
```

#### Scoring

```bash
export S3PRL_DIR=/YOUR/S3PRL/PATH
cd $CORPORA_DIR/quesst14Database/scoring

# dev
./score-TWV-Cnxe.sh $S3PRL_DIR/result/downstream/ExpName_dev \
    groundtruth_quesst14_dev -10

# test
./score-TWV-Cnxe.sh $S3PRL_DIR/result/downstream/ExpName_test \
    groundtruth_quesst14_eval -10
```

## IC: Intent Classification - Fluent Speech Commands

Specified by the command `-d fluent_commands`

#### Prepare data

1. Download and unzip data
    - http://fluent.ai:2052/jf8398hf30f0381738rucj3828chfdnchs.tar.gz

2. Check the prepared file structure

   ```bash
   fluent_speech_commands_dataset
   ├── wavs
   │   └── speakers
   ├── data
   │   └── [*.csv]
   ├── readme.md
   └── Fluent Speech Commands Public License.pdf
   ```

3. Change the following paths under `downstream/fluent_commands/config.yaml` to your own:

   ```yaml
   downstream_expert:
       datarc:
           file_path: "root directory of fluent_speech_commands_dataset"
   ```

#### Training

```bash
python3 run_downstream.py -n ExpName -m train -u fbank -d fluent_commands
```

#### Testing

The testing is done on-the-fly with training since it is not costly. Use the following command to get the testing result from the best-dev ckpt.

```bash
python3 utility/get_best_dev.py result/downstream/ExpName/log.log
```

## SF: End-to-end Slot Filling

#### Prepare data

0. **Optional:** Preprocess Audio SNIPS from the [official version](https://github.com/aws-samples/aws-lex-noisy-spoken-language-understanding).

    ```bash
    # Official Audio SNIPS is in mp3 format, we will convert them to wav
    # We need mp3 support on sox package (originally not supported)
    # First ensure you have the sox installed
    # Then install the mp3 support

    # apt-get
    apt-get install libsox-fmt-mp3

    # or yum install
    yum install soxr sox-plugins-freeworld -y

    # after installing the mp3 support
    CORPORA_DIR="the root directory of all your datasets"
    ./preprocess/snips_prepare_data.sh $CORPORA_DIR
    ```

1. Download the preprocessed Audio SNIPS and unzip
    - https://drive.google.com/file/d/1oBRZd-PaCKz5iY3eZkXs5OB_ZZ4w7bbG/view?usp=sharing

2. Change the paths in `downstream/ctc/snips.yaml`

    ```yaml
    downstream_expert:
        corpus:
            path: "CORPORA_DIR/SNIPS"
        text:
            slots_file: "CORPORA_DIR/SNIPS/slots.txt"
    ```

#### Train

```bash
python3 run_downstream.py -n ExpName -m train -u fbank -d ctc -c downstream/ctc/snips.yaml
```

#### Test

```bash
python3 run_downstream.py -m evaluate -e result/downstream/ExpName/dev-best.ckpt
```

## SID: Speaker Identification

#### Prepare data

1. Download dataset from [Voxceleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) and unzip them.

    ```bash
    voxceleb1_root="/CORPORA_DIR/VoxCeleb1/"
    mkdir -p $voxceleb1_root/dev
    mkdir -p $voxceleb1_root/test
    
    # prepare dev
    cd $voxceleb1_root/dev/
    wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partaa
    wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partab
    wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partac
    wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partad
    cat vox1_dev* > vox1_dev_wav.zip
    unzip vox1_dev_wav.zip

    # prepare test
    cd $voxceleb1_root/test/
    wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_test_wav.zip
    unzip vox1_test_wav.zip
    ```

2. Check prepared file structure

    ```bash
    Voxceleb1/
    ├── dev/
    │   └── wav/
    │       └──Speaker id folders
    └── test/
        └── wav/
            └──Speaker id folders
    ```

3. Change the path in `downstream/voxceleb1/config.yaml`

    ```yaml
    downstream_expert:
        datarc:
            file_path: "root directory of VoxCeleb1"    
    ```

#### Train

```bash
python3 run_downstream.py -n ExpName -m train -u fbank -d voxceleb1
```

#### Test

The testing is done on-the-fly with training since it is not costly. Use the following command to get the testing result from the best-dev ckpt.

```bash
python3 utility/get_best_dev.py result/downstream/ExpName/log.log
```

## ASV: Automatic Speaker Verification

#### Prepare data

1. Follow the step 1 and 2 in **SID**

2. Change the path in `downstream/sv_voxceleb1/config.yaml`

    ```yaml
    downstream_expert:
        datarc:
            file_path: "root directory of VoxCeleb1"    
    ```

#### Training

```bash
python3 run_downstream.py -n ExpName -m train -u fbank -d sv_voxceleb1
```

#### Testing

As there is no official validation set, we save checkpoints every 20000 updates and report the best EER. Evaluating checkpoints take long time so we don't test them on-the-fly on the same GPU. We opt to save all checkpoints and test them parallely with another GPU during training. The following command will run a for-loop to monitor if any new checkpoints is saved, and evaluate it if any is found. The already evaluated checkpoints will be passed as they have the result loggings under their **expdir**.

```bash
./run_while.sh "./downstream/sv_voxceleb1/test_expdirs.sh result/downstream/ExpName; sleep 1800;"
```
#### Report numbers

The lowest number should be reported, which should be at the bottom.

```bash
./downstream/sv_voxceleb1/report.sh result/downstream/ExpName
```

## SD: Speaker Diarization

#### Prepare data

1. Simulate Libri2Mix Data for Diarization

    ```bash
    S3PRL_DIR="root directory of your cloned s3prl"
    CORPORA_DIR"root directory of all your datasets, which hopefully contains LibriSpeech (not necessary)"

    git clone https://github.com/ftshijt/LibriMix.git
    cd LibriMix
    bash generate_librimix.sh $CORPORA_DIR
    python3 scripts/prepare_diarization.py \
        --target_dir $S3PRL_DIR/downstream/diarization/data \
        --source_dir $CORPORA_DIR/Libri2Mix/wav16k/max/metadata
    ```

#### Train

```bash
python3 run_downstream.py -n ExpName -m train -u fbank -d diarization
```

#### Test

```bash
python3 run_downstream.py -m evaluate -e result/downstream/ExpName/best-states-dev.ckpt
```

#### Scoring

1. Clone **dscore**
    
    ```bash
    git clone https://github.com/ftshijt/dscore
    ```

2. Change the path in `downstream/diarization/score.sh`

    ```bash
    dscore_dir="root directory of your cloned dscore"
    ```

3. Run scoring

   ```bash
   ./downstream/diarization/score.sh result/downstream/ExpName downstream/diarization/data/test
   ```

4. The scoring results will look like

   ![](https://i.imgur.com/GnVlFlH.png)

   One should report the lowest number at the bottom, where the column represents DER and the most bottom row will always have the lowest DER which is the number we will report.

5. Re-check the scoring results

    Running the above scoring script takes time. If you want to re-check the scored results, use

    ```bash
    ./downstream/diarization/report.sh result/downstream/ExpName
    ```

## ER: Emotion Recognition

#### Prepare data

1. Download dataset and unzip. You will need to fill a form in IEMOCAP official website to get the dataset.
    - https://sail.usc.edu/iemocap/

2. Preprocess

    ```bash
    python3 ./downstream/emotion/IEMOCAP_preprocess.py "/path/to/IEMOCAP"
    ```

3. Change the path in `downstream/emotion/config.yaml`
    ```yaml
    downstream_expert:
        datarc:
            root: "root directory of IEMOCAP"
    ```

#### Train
IEMOCAP provides 5 splits of data: Section1, Section2, Section3, Section4 and Section5. Conventionally, each split will be selected as the test set and train the model with other 4 splits. That is, 5 times of training and testing is required, and 5 testing scores will be averaged to report the final number. We can use `-v` option to control which split we want to reserve as the test set.

```bash
# -v: fold1, fold2, fold3, fold4, fold5
python3 run_downstream.py -n ExpName -m train -u fbank -d emotion -v fold1
```

#### Test

The testing is done on-the-fly with training since it is not costly. Use the following command to get the testing result from the best-dev ckpt.

```bash
python3 utility/get_best_dev.py result/downstream/ExpName/log.log
```

#### Cross validation

```bash
for test_fold in fold1 fold2 fold3 fold4 fold5;
do
    python3 run_downstream.py -n ExpName_$test_fold -m train -u fbank -d emotion -v $test_fold
    python3 utility/get_best_dev.py result/downstream/ExpName_$test_fold/log.log
done
```

# More tasks

## Phone Classification

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

3. unzip phone labels:

    ```bash
    cd data/cpc_phone
    unzip converted_aligned_phones.zip
    ```

4.  *(Optional)* Allow bucketing to increase training efficientcy & speed, this will generate a directory called `data/len_for_bucket`:

    ```bash
    python preprocess/generate_len_for_bucket.py --data_root "your_libri_root" --output_path ./data/
    ```

5. Change the following paths under `phone_*/config.yaml` to your own:

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

### Intent Classification - SNIPS
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

### Intent Classification - ATIS
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

### Spoken sentiment analysis -  CMU-MOSEI
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

## Source Separation
- **Data preparation**: Simulate Libri2Mix data for source separation. For source separation, we only need 16kHz and min condition. (Usually for source separation, people are using 8kHz min condition, but due to the constrait of pre-trained models we are using 16kHz)

```bash
# download the script and simulate Libri2Mix dataset
git clone https://github.com/HuangZiliAndy/LibriMix.git
cd LibriMix 
./generate_librimix.sh storage_dir

# prepare train, dev and test data in Kaldi format
python downstream/separation_stft/scripts/LibriMix/data_prepare.py \
--part train-100 storage_dir downstream/separation_stft/data

python downstream/separation_stft/scripts/LibriMix/data_prepare.py \
--part dev storage_dir downstream/separation_stft/data

python downstream/separation_stft/scripts/LibriMix/data_prepare.py \
--part test storage_dir downstream/separation_stft/data
```

- **train**:

Train with STFT magnitude as the upstream.

```bash
python3 run_downstream.py \
       --mode train --config downstream/separation_stft/configs/cfg.yaml \
       --downstream separation_stft \
       --upstream stft_mag \
       --upstream_model_config 'upstream/log_stft/stft_mag.yaml' \
       --expdir experiment/separation_stft/stft_mag
```

Train with wav2vec2 as the upstream.

```bash
python3 run_downstream.py \
       --mode train --config downstream/separation_stft/configs/cfg.yaml \
       --downstream separation_stft \
       --upstream wav2vec2 \
       --expdir experiment/separation_stft/wav2vec2
```

I included one upstream called stft_mag in my code, and it is simply extracting STFT magnitude. I notice that s3prl has support for different acoustic features in baseline, but since I am predicting STFT masks, I have to make sure the setup for STFT features and desired STFT masks are identical. 

In other words, (1) when you are using STFT magnitude as the upstream, you need to make sure that the STFT parameters in downstream/separation_stft/configs/cfg.yaml and upstream/log_stft/stft_mag.yaml are identical. (2) When you are using other upstreams like wav2vec2, you need to make sure that the hop_length in downstream/separation_stft/configs/cfg.yaml is the same as the upstream. (like in this file, I am using a hop_length of 320 corresponding to 20ms stride for wav2vec2)

- **test**:

```bash
python3 run_downstream.py \
       --mode evaluate \
       --past_exp experiment/separation_stft/stft_mag/modelbest.ckpt \
       --config downstream/separation_stft/configs/cfg.yaml \
       --downstream separation_stft \
       --upstream stft_mag \
       --upstream_model_config 'upstream/log_stft/stft_mag.yaml' \
       --expdir experiment/separation_stft/stft_mag
```

The model is expected to output si-sdri on the test set.


# Add new downstream tasks

Each downstream task is defined by a **self-contained** folder under this [downstream](./) folder, like the task ASR is defined in [downstream/asr](./asr). Once a new folder is placed under this [downstream](./) folder, says `downstream/blabla/`, you can specify to run this new downstream task with `-d blabla` option in [run_downstream.py](../run_downstream.py) script.

By **self-contained** we mean there should be all the downstream specific materials under your task folder, including the definition for dataset, datalader,model, and loss. How to define these materials are completely free, while the only requirement is to provide an `expert.py` file with an `DownstreamExpert` **nn.module** at the root of your downstream folder, where 3 object methods are implemented: `get_dataloader`, `forward`, and `log_records`.

The fastest way to know how the framework works is to run a minimum example, so we provide a pseudo task [downstream/example/](./example/), which can always be ran up by:

```bash
python3 run_downstream.py -u fbank -d example -n HelloWorld
```

Hence, you can refer to [downstream/example/expert.py](./example/expert.py) for the minimum requirement and implementation specification. Also, you can use [downstream/example/](./example/) as an initial template for developing a new downstream task.

#### Note 1

Please use _relative import_ in your downstream folder, in case we might want to rename or move the location for the `downstream` folder in future.

#### Note 2

If you want to train your downstream task with distributed training, you should take care to use **DistributedSampler** when providing the training dataloader in your expert file.
