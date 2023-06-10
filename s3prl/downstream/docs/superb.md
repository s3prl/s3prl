# SUPERB Benchmark & Challenge

## Prerequisite

Please read [downstream/README.md](../README.md) for the general command pattern, and read [upstream/example/README.md](../../upstream/example/README.md) for registering a new pretrained model (upstream).

## Introduction

In this document we detail the commands for reproducing the paper [**SUPERB:** **S**peech processing **U**niversal **PER**formance **B**enchmark](https://arxiv.org/abs/2105.01051) and [**SUPERB-SG:** Enhanced **S**peech processing **U**niversal **PER**formance
**B**enchmark for **S**emantic and **G**enerative Capabilities](https://arxiv.org/abs/2203.06849). If you use the tasks here for your research, please consider citing the following papers:

```
@inproceedings{yang21c_interspeech,
  author={Shu-wen Yang and Po-Han Chi and Yung-Sung Chuang and Cheng-I Jeff Lai and Kushal Lakhotia and Yist Y. Lin and Andy T. Liu and Jiatong Shi and Xuankai Chang and Guan-Ting Lin and Tzu-Hsien Huang and Wei-Cheng Tseng and Ko-tik Lee and Da-Rong Liu and Zili Huang and Shuyan Dong and Shang-Wen Li and Shinji Watanabe and Abdelrahman Mohamed and Hung-yi Lee},
  title={{SUPERB: Speech Processing Universal PERformance Benchmark}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
  pages={1194--1198},
  doi={10.21437/Interspeech.2021-1775}
}
```
```
@article{superb_sg,
  title={SUPERB-SG: Enhanced Speech processing Universal PERformance Benchmark for Semantic and Generative Capabilities},
  author={Hsiang-Sheng Tsai and Heng-Jui Chang and Wen-Chin Huang and Zili Huang and Kushal Lakhotia and Shu-wen Yang and Shuyan Dong and Andy T. Liu and Cheng-I Lai and Jiatong Shi and Xuankai Chang and Phil Hall and Hsuan-Jui Chen and Shang-Wen Li and Shinji Watanabe and Abdel-rahman Mohamed and Hung-yi Lee},
  journal={ArXiv},
  year={2022},
  volume={abs/2203.06849}
}
```
Besides the tasks presented in the paper, we are also extending the coverage over all speech tasks. In the [SUPERB Challenge](https://superbbenchmark.org/challenge) in [*AAAI workshop: The 2nd Self-supervised Learning for Audio and Speech Processing*](https://aaai-sas-2022.github.io/), more tasks are introduced into the benchmark framework, and the setup detailed here serves as the **public-set** in the challenge. We list all tasks below:

| ID | Task Name | Category | Paper | Challenge public-set |
| - | - | - | - | - |
| [PR](#pr-phoneme-recognition) | Phoneme Recognition | Content | V | V |
| [ASR](#asr-automatic-speech-recognition) | Automatic Speech Recognition | Content | V | V |
| [KS](#ks-keyword-spotting) | Keyword Spotting | Content | V |  |
| [QbE](#qbe-query-by-example-spoken-term-detection) | Query-by-Example | Content | V | V |
| [SID](#sid-speaker-identification) | Speaker Identification | Speaker | V | V |
| [ASV](#asv-automatic-speaker-verification) | Automatic Speaker Verification | Speaker | V | V |
| [SD](#sd-speaker-diarization) | Speaker Diarization | Speaker | V | V |
| [ER](#er-emotion-recognition) | Emotion Recognition | Paralinguistics | V | V |
| [IC](#ic-intent-classification) | Spoken Intent Classification | Semantics | V | |
| [SF](#sf-end-to-end-slot-filling) | Spoken Slot Filling | Semantics | V |  |
| [ST](#st-speech-translation) | Speech Translation | Semantics | V | V |
| [SE](#se-speech-enhancement) | Speech Enhancement | Generation | V | V |
| [SS](#ss-source-separation) | Source Separation | Generation | V | V |
| [VC](#vc-voice-conversion) | Voice Conversion | Generation | V |  |

This document contains the following meterials:

#### [The command for each task](#task-specific-usages)

- Data preparation
- Training
- Testing / Scoring

#### [The training artifacts of each task](./superb_artifacts.md)

- Tensorboard logs
- Trained downstream weights (the best on dev set)

#### [Leaderboard submission helper](#leaderboard-submission)

- Ready for the tasks presented in the paper
- Will be ready for the challenge on **Sep 30, 2021**
    - New tasks submission
    - Overall metrics

# Task-specific usages

To reproduce the results in the SUPERB paper, you can follow the commands below by only changing the learning rate: `config.optimizer.lr` in the config file with the `override` option.

```bash
# The default lr for ASR is 1.0e-4
python3 run_downstream.py -m train -u wav2vec2 -d asr -n ExpName \
    -o config.optimizer.lr=1.0e-5
```

If the fully converged training time is too long, you can also consider using [distributed training](../README.md#distributed-training) to avoid the gradient accumulation.

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
Installing all the dependencies right could be quite complicated. Note that the decoding is not required for SSL representations to perform well on ASR and you can also skip the ASR results from LM decoding when submitting to the leaderboard.

##### I. Prepare Decoding Environment

1. Install [KenLM](https://github.com/kpu/kenlm)
    - Please follow the official installation instructions of KenLM instead of the one documented in flashlight or wav2letter du to some known issues.

2. Install [flashlight python bindings](https://github.com/flashlight/flashlight/blob/master/bindings/python/README.md)
    - Only the **python bindings** is required instead of the entire flashlight toolkit

3. Download LibriSpeech official 4-gram LM
    - https://www.openslr.org/resources/11/4-gram.arpa.gz
    - Downloaded filename: **4-gram.arpa.gz**

4. Download character-based lexicon
    - https://dl.fbaipublicfiles.com/fairseq/wav2vec/librispeech_lexicon.lst
    - Downloaded filename: **librispeech_lexicon.lst**

5. Make sure your fairseq version contains this commit [cb8469](https://github.com/pytorch/fairseq/commit/cb84694c195afced474d17318b5e746d1a9d20a3#diff-ee3a94b6d9b5f2cc60f1b69afc075abbe2061083b52515178eb7145d59e7e7e4)

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

```bash
python3 run_downstream.py -m evaluate -e result/downstream/ExpName/dev-best.ckpt
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

In SUPERB, we run DTW for all the hidden states layer-by-layer. Choose the best layer according to dev set and report its score on the test set. A specific layer can be selected by `-l` option, indexed from 0. The following take the last layer as an example.

```bash
# The default dist_fn if not specified is "cosine_exp"
# as it yields the best result for almost all upstream
# Supported dist_fn: cosine, cityblock, euclidean, cosine_exp

layer=-1;
dist_fn=cosine;

# dev
python3 run_downstream.py -m evaluate -t "dev" -u hubert -l ${layer} \
    -d quesst14_dtw -n ExpName_${layer}_dev \
    -o config.downstream_expert.dtwrc.dist_method=$dist_fn

# test
python3 run_downstream.py -m evaluate -t "test" -u fbank -l ${layer} \
    -d quesst14_dtw -n ExpName_${layer}_test \
    -o config.downstream_expert.dtwrc.dist_method=$dist_fn
```

#### Scoring

```bash
export S3PRL_DIR=/YOUR/S3PRL/PATH
cd $CORPORA_DIR/quesst14Database/scoring

# dev
./score-TWV-Cnxe.sh $S3PRL_DIR/result/downstream/ExpName_${layer}_dev \
    groundtruth_quesst14_dev -10

# test
./score-TWV-Cnxe.sh $S3PRL_DIR/result/downstream/ExpName_${layer}_test \
    groundtruth_quesst14_eval -10
```

#### Submit

After you benchmark all the layers of an upstream, says you find the 6-th layer is the best for QbE according to dev set. Please use `ExpName_6_test` as the submission expdir for [`submit.py`](../../submit/submit.py).

## IC: Intent Classification

Specified by the command `-d fluent_commands`

#### Prepare data

1. Download and unzip data: Fluent Speech Commands
    - Official data link: http://fluent.ai:2052/jf8398hf30f0381738rucj3828chfdnchs.tar.gz
    - Official website: https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/
    - Since the official link might break occasionally, we provide a backup link. If this is not allowed please let us know and we will remove it immediately.
    - Please use `wget http://140.112.21.28:9000/fluent.tar.gz`

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

```bash
python3 run_downstream.py -m evaluate -e result/downstream/ExpName/dev-best.ckpt
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

#### Training

```bash
python3 run_downstream.py -n ExpName -m train -u fbank -d ctc -c downstream/ctc/snips.yaml
```

#### Testing

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

#### Training

```bash
python3 run_downstream.py -n ExpName -m train -u fbank -d voxceleb1
```

#### Testing

```bash
python3 run_downstream.py -m evaluate -e result/downstream/ExpName/dev-best.ckpt
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

If you already know a specific checkpoint to test, says ***states-20000.ckpt***, you can test it with:

```bash
python3 run_downstream.py -m evaluate -e result/downstream/ExpName/states-20000.ckpt
```

However, there is no official validation set under VoxCeleb1 setting, we save checkpoints every 20000 updates and report the best EER. Evaluating checkpoints take long time so we don't test them along with training on a single GPU. We save all checkpoints and test them parallely with another GPU. The following command will:

1. Run a for-loop to find newly saved checkpoints in *expdir*
2. Evaluate it if any is found and log the testing result
3. Prepare the best prediction file according to already tested checkpoints

Note. The already evaluated checkpoints will be passed.

```bash
voxceleb1="root directory of VoxCeleb1"
./downstream/sv_voxceleb1/test_expdir.sh result/downstream/ExpName $voxceleb1
```

## SD: Speaker Diarization

We prepare the frame-wise training label on-the-fly, and convert the frame-wise prediction into RTTM files annotated in seconds. The inferenced RTTM will then be scored by comparing to the groundtruth RTTM by [dscore](https://github.com/ftshijt/dscore). You can choose the `frame_shift` (stride) of the training label for the upstream representation. This only affects the training materials and does not affect the groundtruth RTTM, which is fixed in [Libri2Mix](https://github.com/s3prl/LibriMix) during data preparation.

#### Prepare data

Simulate Libri2Mix Data for Diarization

```bash
S3PRL_DIR="root directory of your cloned s3prl"
CORPORA_DIR"root directory of all your datasets, which hopefully contains LibriSpeech (not necessary)"

git clone https://github.com/s3prl/LibriMix.git
cd LibriMix
bash generate_librimix_sd.sh $CORPORA_DIR
python3 scripts/prepare_diarization.py \
    --target_dir $S3PRL_DIR/downstream/diarization/data \
    --source_dir $CORPORA_DIR/Libri2Mix/wav16k/max/metadata
```

#### Training

Train with the label in the same `frame_shift` as the upstream representation: (**recommened**)

```bash
python3 run_downstream.py -n ExpName -m train -u fbank -d diarization
```

Train with the label in a specific `frame_shift` (e.g. 160):

```bash
python3 run_downstream.py -n ExpName -m train -u fbank -d diarization \
    -o config.downstream_expert.datarc.frame_shift=160
```

The upstream representation will be upsampled (duplicate) or downsampled (take 1 per N frames) to match the sequence length of your assigned label. This can be useful when the representation has too small `frame_shift` and hence too long sequence, which leads to too long training time.

#### Testing

The `frame_shift` for the training label is already saved in the checkpoint, and the same `frame_shift` will be used to convert the frame-wise prediction into RTTM files annotated in seconds.

##### I. Inference predictions (for submission and for scoring locally)

```bash
python3 run_downstream.py -m evaluate -e result/downstream/ExpName/best-states-dev.ckpt
```

##### II. Scoring (not required for submission)

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

5. Re-check the scoring results: Running the above scoring script takes time. If you want to re-check the scored results, use

    ```bash
    ./downstream/diarization/report.sh result/downstream/ExpName
    ```

## ER: Emotion Recognition

#### Prepare data

1. Download dataset and unzip. You will need to fill a form in IEMOCAP official website to get the dataset.
    - https://sail.usc.edu/iemocap/

2. Change the path in `downstream/emotion/config.yaml`
    ```yaml
    downstream_expert:
        datarc:
            root: "root directory of IEMOCAP"
    ```

#### Training

IEMOCAP provides 5 splits of data: Section1, Section2, Section3, Section4 and Section5. Conventionally, each split will be selected as the test set and train the model with other 4 splits. That is, 5 times of training and testing is required, and 5 testing scores will be averaged to report the final number. We can change the `test_fold` option in the config file to control which split we want to reserve as the test set.

```bash
# test_fold can be: fold1, fold2, fold3, fold4, fold5
python3 run_downstream.py -n ExpName -m train -u fbank -d emotion -c downstream/emotion/config.yaml -o "config.downstream_expert.datarc.test_fold='fold1'"
```

#### Testing

```bash
python3 run_downstream.py -m evaluate -e result/downstream/ExpName/dev-best.ckpt
```

#### Cross validation

```bash
for test_fold in fold1 fold2 fold3 fold4 fold5;
do
    # The default config is "downstream/emotion/config.yaml"
    python3 run_downstream.py -n ExpName_$test_fold -m train -u fbank -d emotion -o "config.downstream_expert.datarc.test_fold='$test_fold'"
    python3 run_downstream.py -m evaluate -e result/downstream/ExpName_$test_fold/dev-best.ckpt
done
```

## SS: Source Separation

We have two versions for the source separation task. The first version (separation_stft) is the same as the SUPERB-SG paper. In the second version (separation_stft2), we further improve the system performance and training speed. Please refer to [README](../separation_stft2/README.md) about the detailed changes of the second version.

#### Prepare data

Simulate Libri2Mix data for source separation. For source separation, we only need 16kHz and min condition.

```bash
# Download the script and simulate Libri2Mix dataset
git clone https://github.com/s3prl/LibriMix.git
cd LibriMix 
./generate_librimix_ss.sh storage_dir

# Prepare train, dev and test data in Kaldi format.
# Replace separation_stft with separation_stft2 if
# you are using the second version
python downstream/separation_stft/scripts/LibriMix/data_prepare.py \
--part train-100 storage_dir/Libri2Mix downstream/separation_stft/datasets/Libri2Mix

python downstream/separation_stft/scripts/LibriMix/data_prepare.py \
--part dev storage_dir/Libri2Mix downstream/separation_stft/datasets/Libri2Mix

python downstream/separation_stft/scripts/LibriMix/data_prepare.py \
--part test storage_dir/Libri2Mix downstream/separation_stft/datasets/Libri2Mix

# Subsample dev set from 3000 utts to 1000 utts (for faster validation)
# This step is not necessary for the second version since the training
# and validation speed is much faster
python downstream/separation_stft/scripts/LibriMix/subsample.py \
downstream/separation_stft/datasets/Libri2Mix/wav16k/min/dev \
downstream/separation_stft/datasets/Libri2Mix/wav16k/min/dev_1000

cd $YOUR_S3PRL_ROOT/s3prl/
```

#### Training

Train with STFT magnitude as the upstream. The default stride is 20ms, and you can adjust that in `upstream/log_stft/stft_mag.yaml`. Replace separation_stft with separation_stft2 if you are using the second version.

```bash
python3 run_downstream.py -m train \
        -d separation_stft \
        -c downstream/separation_stft/configs/cfg.yaml \
        -u stft_mag \
        -g 'upstream/log_stft/stft_mag.yaml' \
        -n ExpName
```

Train with wav2vec2 as the upstream.

```bash
python3 run_downstream.py --mode train \
        -d separation_stft \
        -c downstream/separation_stft/configs/cfg.yaml \
        -u wav2vec2 \
        -n ExpName \
```

#### Testing

```bash
python3 run_downstream.py -m evaluate \
        -e result/downstream/ExpName/best-states-dev.ckpt \
```

The model is expected to output SI-SDRi on the test set.

## SE: Speech Enhancement

To be comparable to SUPERB benchmark, please follow the `downstream/enhancement_stft` folder.
We have a second version in the `downstream/enhancement_stft2` folder, which gets improved speech enhancement performance with SSL features. However, the second version is not comparable to the results on the SUPERB benchmark, but a recipe helpful for people more interested in boosting the performance for speech enhancement.

#### Prepare data

1. We use Voicebank-DEMAND dataset for speech enhancement. We follow the data preparation steps in SpeechBrain:

    ```bash
    # Download the Voicebank-DEMAND dataset and convert it to 16kHz
    # I am following the data preparation script in SpeechBrain toolkit (https://github.com/speechbrain/speechbrain/blob/develop/recipes/Voicebank/voicebank_prepare.py)
    from voicebank_prepare import download_vctk
    download_vctk(data_dir)
    ```

    However, the above pipeline might take too much time to download the original dataset. Hence, we also provide the already preprocessed archive:

    ```bash
    wget http://140.112.21.28:9000/noisy-vctk-16k.zip
    unzip noisy-vctk-16k.zip
    ```

2. Check the unzipped voicebank directory structure

    ```bash
    data_dir/
    ├── clean_testset_wav_16k/
    ├── clean_trainset_28spk_wav_16k/
    ├── noisy_testset_wav_16k/
    ├── noisy_trainset_28spk_wav_16k/
    ├── testset_txt/
    └── trainset_28spk_txt/
    ```

3. Prepare kaldi-style scp files. Replace enhancement_stft with enhancement_stft2
   if you are using the second version.

    ```bash
    # prepare train, dev and test data in Kaldi format
    python downstream/enhancement_stft/scripts/Voicebank/data_prepare.py \
        data_dir downstream/enhancement_stft/datasets/voicebank --part train
    python downstream/enhancement_stft/scripts/Voicebank/data_prepare.py \
        data_dir downstream/enhancement_stft/datasets/voicebank --part dev
    python downstream/enhancement_stft/scripts/Voicebank/data_prepare.py \
        data_dir downstream/enhancement_stft/datasets/voicebank --part test
    ```

#### Training

Train with hubert as the upstream.

```bash
python3 run_downstream.py -m train \
       -c downstream/enhancement_stft/configs/cfg_voicebank.yaml \
       -d enhancement_stft \
       -u hubert \
       -n ExpName \
```

#### Testing

```bash
python3 run_downstream.py -m evaluate \
       -e result/downstream/ExpName/best-states-dev.ckpt \
```
The model is expected to output PESQ, STOI, and SI-SDRi on the test set.

## VC: Voice conversion

The following instruction is only a minimal description for benchmarking. A complete guide about the task, dataset, implementation and usage can be found in the [README](../a2o-vc-vcc2020/README.md). We evaluate the VC capability by training 4 target speaker models that given any source speaker utterance, the single-speaker model can convert it to a specific target speaker. This setting is known as Any-to-one VC. The trained 4 target speakers are: TEF1, TEF2, TEM1, TEM2. The quality of the target speaker model is evaluated with MCD (lower better). One should average the MCD from four speakers.

#### Prepare data

Download the VCC2020 dataset and the pretrained vocoder.

```
cd downstream/a2o-vc-vcc2020
cd data
./data_download.sh vcc2020/
cd ../

# Download the pretrained PWGs.
./vocoder_download.sh ./
```

#### Training

Specify a target speaker for training from: TEF1, TEF2, TEM1, TEM2

```
python run_downstream.py -m train -n EXPNAME -u wav2vec -d a2o-vc-vcc2020 \
    -o config.downstream_expert.trgspk=TEF1
```

#### Testing

Waveform generation and evaluation (using wav2vec for example) for a specific checkpoint.

```
./downstream/a2o-vc-vcc2020/decode.sh ./downstream/a2o-vc-vcc2020/pwg_task1 result/downstream/EXPNAME/<step> TEF1
```

## ST: Speech Translation

The following instruction is only a minimal description for benchmarking. A complete guide about the task, dataset, implementation and usage can be found in the [README](../speech_translation/README.md).

#### Prepare data

Preparing CoVoST En->De dataset.

1. Download [Common Voice audio clips and transcripts (english)](https://commonvoice.mozilla.org/en/datasets) (Common Voice Corpus 4).

2. Change the path in `downstream/speech_translation/prepare_data/prepare_covo.sh`

```bash
covo_root="root directory of covost"
src_lang=en
tgt_lang=de
```

3. Run the following script

```bash
cd downstream/speech_translation/prepare_data/
bash prepare_covo.sh
```

#### Training

```
python run_downstream.py -m train -n ExpName -u fbank -d speech_translation
```

#### Testing

```
python run_downstream.py -m evaluate -e result/downstream/ExpName/dev-best.ckpt
```

The model will report case-sensitive detokenized BLEU.

## OOD-ASR: Out-of-domain Automatic Speech Recognition Tasks

Read [README](../ctc/README.md).


# Leaderboard submission

After *finishing the **Testing*** of each task, the prediction files for leaderboard submission will be located under the `expdir`. You can use [submit.py](../../submit/submit.py) to easily organize them into a zip file which can later be submitted to our [leaderboard](https://superbbenchmark.org/submit). We now support submissions for the following tasks: **PR**, **ASR**, **KS**, **QbE**, **SID**, **ASV**, **SD**, **IC**, **SF**, **ER**, **SE**, **SS**, **ST**.

If you find [superbbenchmark.org](www.superbbenchmark.org) is down temporarily, please try to use [140.112.21.28](140.112.21.28) as an alternative. They share the same backend. We will make the official domain work as soon as possible.

Please use the master branch newer than [852db2e](https://github.com/s3prl/s3prl/commit/852db2e5f65fc9baea4a5877ffda6dd7470c72fc). Note that our SUPERB codebase is backward-compatible, so you don't need to re-train any model after upgrading to this newer version. You only need this new version to inference the prediction files for submission correctly.

```sh
output_dir="submission"

python3 submit/submit.py \
    --output_dir $output_dir \
    --pr pr_expdir \
    --sid sid_expdir \
    --ks ks_expdir \
    --ic ic_expdir \
    --er_fold1 er_fold1_expdir \
    --er_fold2 er_fold2_expdir \
    --er_fold3 er_fold3_expdir \
    --er_fold4 er_fold4_expdir \
    --er_fold5 er_fold5_expdir \
    --asr_no_lm asr_expdir \
    --asr_with_lm asr_expdir \
    --qbe qbe_expdir \
    --sf sf_expdir \
    --sv sv_expdir \
    --sd sd_expdir \
    --se se_expdir \
    --ss ss_expdir \
    --st st_expdir \
```

After executing, you can submit **submission/predict.zip** to the leaderboard.

We also prepare the [**example-expdirs**](https://superbbenchmark.org/api/download/expdirs) for you to diagnose if the submission fails. After unzipping you will see the following structure:

```sh
expdirs/
    asr_expdir/
    er_fold1_expdir/
    er_fold2_expdir/
    er_fold3_expdir/
    er_fold4_expdir/
    er_fold5_expdir/
    ic_expdir/
    ks_expdir/
    pr_expdir/
    qbe_expdir/
    ...
```

Each **expdir** will contain the minimal submission-related files which should also appear in your **expdir** after you do the testing. Here is an [**example-script**](../submit/demo_submit.sh) on how to use the above **example-expdirs** to prepare a submittable zip file.

```sh
cd s3prl/s3prl/submit
./demo_submit.sh examples
```

After executing, you will see:

```sh
s3prl/s3prl/submit/examples/
    expdirs/
    expdirs.zip
    predict/
    predict.zip
```

The [**predict.zip**](https://superbbenchmark.org/api/download/example) is the one for you to submit.

##### Note1
You don't need to prepare all the **expdirs** for the submission. You can zip only a subset of **expdirs**. After your submission, the leaderboard will only show the results of your submitted tasks. Eg.

```sh
python3 submit/submit.py \
    --output_dir submission \
    --pr pr_expdir
```

The above command will produce a **predict.zip** which will only show the PR score after submitted to the leaderboard.

##### Note2
Emotion Recognition (er) does 5-fold cross validation: 5 training and 5 testing, so 5 **expdirs** in total.

##### Note3
The **expdirs** for `asr_no_lm` and `asr_with_lm` are typically the same. Since the same ASR downstream model was trained and just decoded in different ways, so the same **expdir** assigned for training is used when testing. The default testing will produce predictions for `asr_no_lm`. By using Kenlm decoding you can get predictions for `asr_with_lm`. See ASR section below for more information.
