# Downstream tasks

Self-supervised (SSL) pretrained models are not able to justify their effectiveness through pretraining loss. One always has to evaluate their performance with downstream tasks. Hence, it is crucial to collect a wide range of downstream tasks and make the evaluation pipeline as easy as possible to speed up the development cycle.

Basing on the tasks already developed, we further select several representative ones to form the [**Benchmark**](#benchmark) for SSL models.

If you are interested in developing your own task, please check [**Add new downstream tasks**](#add-new-downstream-tasks). Pull requests are always welcome.

# Benchmark

The toolkit supports the following benchmark. To benchmark your pretrained model please follow these sections:

1. [**Clone the repository and install packages**](../README.md#installation)
2. [**Setup your customized upstream model**](TBD)
3. Follow the benchmark-specific instructions described below.

### **SUPERB:** **S**peech processing **U**niversal **PER**formance **B**enchmark

1. Follow the task-specific instructions in: [**PR**](#pr-phoneme-recognition), [**KS**](#ks-keyword-spotting), [**IC**](#ic-intent-classification---fluent-speech-commands), [**SID**](#sid-speaker-identification), [**ER**](#er-emotion-recognition), [**ASR**](#asr-automatic-speech-recognition), [**QbE**](#qbe-query-by-example-spoken-term-detection), [**SF**](#sf-end-to-end-slot-filling---snips), [**ASV**](#asv-speaker-verification), [**SD**](#sd-speaker-diarization)

# General usage

All of the downstream task follow the following command pattern, with a few task-specific adjustments which are detailed in [**Task-specific usage**](#task-specific-usage).

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
  -o "config.optimizer.lr=1.0e-3,,config.optimizer.name=\"AdamW\",,config.runner.eval_dataloaders=\"['dev', 'test']\""
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

We wrap the model with **DistributedDataParallel**. By inserting `-m torch.distributed.launch --nproc_per_node {GPU_NUM}` between `python3` and `run_downstream.py`, you can directly turn the above **training** commands into distributed training. Currently only **ASR** and **SV** support distributed training.

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

Only the training part is powered by **DistributedDataParallel**, and we save all the model _state_dict_ **without** the DDP wrapper. That is, after the DDP training, you can always evaluate the checkpoint using the testing command documented above (on single GPU).

# Task-specific Usage

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

4. _(Optional)_ Allow bucketing to increase training efficientcy & speed, this will generate a directory called `data/len_for_bucket`:

   ```bash
   python3 preprocess/generate_len_for_bucket.py --data_root "your_libri_root" --output_path ./data/
   ```

5. Change the following paths under `phone_*/config.yaml` to your own:

   ```yaml
   libri_root: "/media/andi611/1TBSSD/LibriSpeech/"
   bucket_file: "data/len_for_bucket"
   ```

#### Training

```bash
python3 run_downstream.py -m train -u baseline -d phone_linear -n ExpName
python3 run_downstream.py -m train -u baseline -d phone_linear_concat -n ExpName
python3 run_downstream.py -m train -u baseline -d phone_1hidden -n ExpName
```

#### Testing

Testing is done on-the-fly with training since it is not costly. Use the following command to get the testing result from the best-dev ckpt:

```bash
python3 utility/get_best_dev.py result/downstream/ExpName/log.log
```

## PR: Phoneme Recognition

Specified by the command: `-d ctc`

#### Prepare data

1. Download the raw [LibriSpeech](https://www.openslr.org/12) corpus and unzip, we need

   - train-clean-100
   - dev-clean
   - test-clean

2. Modify `downstream_expert.corpus.path` in `downstream/ctc/libriphone.yaml` with the **root directory** of LibriSpeech.

#### Training

```bash
python3 run_downstream.py -m train -u fbank -d ctc --config downstream/ctc/libriphone.yaml -n ExpName
```

#### Testing

```bash
expdir=result/downstream/ExpName;
python3 run_downstream.py -m evaluate -e $expdir/dev-best.ckpt > $expdir/test.result
```

## ASR: Automatic Speech Recognition

Specified by the command: `-d asr`

#### Prepare data

1. Download the raw [LibriSpeech](https://www.openslr.org/12) corpus and unzip

   - train-clean-100
   - dev-clean
   - test-clean

2. Modify `downstream_expert.libri_root` in `downstream/asr/config.yaml` with the **root directory** of LibriSpeech.

3. Prepare the lengths for utterances in LibriSpeech's train-clean-100, dev-clean and test-clean:

   ```bash
   # Official LibriSpeech is in .flac format
   python3 preprocess/generate_len_for_bucket.py -i /root/of/LibriSpeech -o data/librispeech -a .flac --n_jobs 12
   ```

#### Prepare Decoding Environment

The current document will report the WER without LM. Detailed usage for the decoding with LM will be updated soon.

<details><summary>Useful links</summary><p>

- Install wav2letter python bindings (not the entire wav2letter is needed to be installed):
- https://github.com/facebookresearch/wav2letter/wiki/Building-Python-bindings
- When installing KenLM, please follow the [official instruction](https://github.com/kpu/kenlm). This is a [known issue](https://github.com/facebookresearch/wav2letter/issues/875).
- If you encounter issue when installing KenLM, you might need to install some [extra dependencies](https://medium.com/tekraze/install-kenlm-binaries-on-ubuntu-language-model-inference-tool-33507000f33).
- LM on fairseq:
- https://github.com/facebookresearch/wav2letter/tree/v0.2/recipes/models/sota/2019
- LibriSpeech Official LM:
- https://www.openslr.org/resources/11/4-gram.arpa.gz
- Lexicon: - https://dl.fbaipublicfiles.com/fairseq/wav2vec/librispeech_lexicon.lst
</p></details>

#### Train

```bash
python3 run_downstream.py -m train -u fbank -d asr -n ExpName
```

#### Test

```bash
expdir=result/downstream/ExpName;
python3 run_downstream.py -m evaluate -t "test-clean" -e $expdir/dev-best.ckpt
```

## KS: Keyword Spotting

Specified by the command: `-d speech_commands`

#### Prepare data

1. You can use either v0.01 or v0.02, and I assume you're using v0.01 in the following steps. **We use V0.01 in SUPERB.**

   - http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz
   - http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
   - http://download.tensorflow.org/data/speech_commands_test_set_v0.01.tar.gz
   - http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz

2. Download and unpack Speech Commands

   ```bash
   mkdir -p /YOUR_CORPORA_DIR/speech_commands_v0.01
   tar zxf speech_commands_v0.01.tar.gz -C /YOUR_CORPORA_DIR/speech_commands_v0.01
   ```

3. Download and unpack Speech Commands test set

   ```bash
   mkdir -p /YOUR_CORPORA_DIR/speech_commands_test_set_v0.01
   tar zxf speech_commands_test_set_v0.01.tar.gz -C /YOUR_CORPORA_DIR/speech_commands_test_set_v0.01
   ```

4. Change the following path in `speech_commands/config.yaml` to yours

   ```yaml
   speech_commands_root: /YOUR_CORPORA_DIR/speech_commands_v0.01
   speech_commands_test_root: /YOUR_CORPORA_DIR/speech_commands_test_set_v0.01
   ```

#### train

```bash
python3 run_downstream.py -m train -u fbank -d speech_commands -n TrainExpName
```

#### test

The testing is done on-the-fly with training since it is not costly. Use the following command to get the testing result from the best-dev ckpt

```bash
python3 utility/get_best_dev.py result/downstream/TrainExpName/log.log
```

## QbE: Query-by-Example Spoken Term Detection

This task does not require training. We extract representations and run dynamic time warping (DTW) on them.

Specified by the command `-d quesst14_dtw`

#### Prepare data

1. Download the data
   - https://speech.fit.vutbr.cz/files/quesst14Database.tgz
2. Specify the place to unpack the database

   ```bash
   export CORPORA_DIR=/YOUR/CORPORA/DIR/PATH
   ```

3. Unpack the tarball

   ```bash
   tar zxf quesst14Database.tgz -C $CORPORA_DIR
   ```

4. Change the following path in `quesst14/config.yaml` to yours
   ```yaml
   dataset_root: /YOUR/CORPORA/DIR/PATH/quesst14Database
   ```

#### Dynamic Time Warping (DTW)

```bash
# The default dist_fn if not specified is "cosine_exp"
# as it yields the best result for almost all upstream
# Supported dist_fn: cosine, cityblock, euclidean, cosine_exp

dist_fn=cosine;
python3 run_downstream.py -m evaluate -t "test" -u fbank \
    -d quesst14_dtw -n fbank_dtw_test \
    -o "config.downstream_expert.dtwrc.dist_method=\"$dist_fn\""
```

#### Scoring

```bash
export S3PRL_DIR=/YOUR/S3PRL/PATH
cd $CORPORA_DIR/quesst14Database/scoring
./score-TWV-Cnxe.sh $S3PRL_DIR/result/downstream/fbank_dtw_test \
    groundtruth_quesst14_eval -10
```

## IC: Intent Classification - Fluent Speech Commands

Specified with the command `-d fluent_commands`

#### Prepare data

1. Download and unzip data to the path you want:

   ```bash
   cd /path/to/put/data
   wget http://fluent.ai:2052/jf8398hf30f0381738rucj3828chfdnchs.tar.gz
   tar zxvf jf8398hf30f0381738rucj3828chfdnchs.tar.gz
   ```

2. After extracting the file, you should have the file structure as following:

   ```bash
   fluent_speech_commands_dataset
   ├── wavs
   │   └── speakers
   ├── data
   │   └── [*.csv]
   ├── readme.md
   └── Fluent Speech Commands Public License.pdf
   ```

3. Change the following paths under `fluent_commands/config.yaml` to your own:

   ```yaml
   file_path: /home/raytz/Disk/data/fluent_speech_commands_dataset
   ```

#### Train

```bash
python3 run_downstream.py -m train -u baseline -d fluent_commands -n TrainExpName
```

#### Test

The testing is done on-the-fly with training since it is not costly. Use the following command to get the testing result from the best-dev ckpt.

```bash
python3 utility/get_best_dev.py result/downstream/TrainExpName/log.log
```

## SF: End-to-end Slot Filling - SNIPS

#### Prepare env for text normalization

```python
import nltk

nltk.download('brown')
nltk.download('names')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
```

#### Data preparation

```bash
git clone https://github.com/s3prl/s3prl.git
git checkout develop
./s3prl/preprocess/snips_prepare_data.sh
```

#### Train

First, change the paths for `downstream_expert.corpus.path` and `downstream_expert.corpus.text.slots_file` in the **downstream/ctc/snips.yaml**

```bash
python3 run_downstream.py -m train -u wav2vec2 -d ctc -c downstream/ctc/snips.yaml -n TrainExpName
```

#### Test

```bash
expdir=result/downstream/[TrainExpName];
python3 run_downstream.py -m evaluate -e $expdir/dev-best.ckpt > $expdir/test.result
```

## SID: Speaker Identification

#### Prepare data

1. Download dataset from [Voxceleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) and unzip them.

2. put dev / test set into one folder (root)

   ```bash
   Voxceleb1(root)
   ├── dev
   │   └── wav
           └──Speaker id folder
   ├── test
   │   └── wav
           └──Speaker id folder
   ```

3. Change the following paths under `./downstream/voxceleb1/config.yaml` to your own:

   ```bash
     downstream_expert:
       datarc:
         file_path: /path/to/VoxCeleb1
   ```

#### Train

```bash
python3 run_downstream.py -m train -d voxceleb1 -u fbank -n TrainExpName
```

#### Test

The testing is done on-the-fly with training since it is not costly. Use the following command to get the testing result from the best-dev ckpt.

```bash
python3 utility/get_best_dev.py result/downstream/TrainExpName/log.log
```

## ASV: Speaker Verification

#### Prepare data

Follow the same pipeline as that in **SID**

#### Train

```bash
python3 run_downstream.py -m train -u apc -d sv_voxceleb1 -n TrainExpName
```

#### Test

Testing stage of ASV takes a really long time, as we have to evaluate 10 checkpoints per upstream (there is not official validation set) and each of them need time (this depends on the speed of upstream inference, for **fbank** only 15 mins but for **pase** is 1.5 hour).

Hence, we opt to save all checkpoints and test them parallely with another GPU during training. The following command will run a for-loop to monitor if any new checkpoints is saved, and evaluate it if any is found. The already evaluated checkpoints will be passed as they have the result loggings under their **expdir**.

```bash
./run_while.sh "./downstream/sv_voxceleb1/test_expdirs.sh result/downstream/TrainExpName; sleep 1800;"
```

#### Report numbers

The lowest number should be reported, which should be at the bottom.

```bash
./downstream/sv_voxceleb1/report.sh result/downstream/TrainExpName
```

## SD: Speaker Diarization

#### Prepare data

Simulate Libri2Mix Data for Diarization

```bash
git clone https://github.com/ftshijt/LibriMix.git
cd LibriMix

# This script need the sox binary installed
./generate_librimix.sh storage_dir

python3 scripts/prepare_diarization.py --target_dir ../downstream/diarization/data
```

#### Train

```bash
python3 run_downstream.py -m train -c ./downstream/diarization/config.yaml -d diarization -u baseline -n libri2mix_diar
```

#### Test

```bash
python3 run_downstream.py -m evaluate -t test -e result/downstream/libri2mix_diar/best-states-dev.ckpt
```

#### Scoring

1. Clone [dscore](https://github.com/ftshijt/dscore)
2. change the dscore_dir (line 13) to your root directory of the cloned dscore in `downstream/diarization/score.sh` and then run

   ```bash
   ./downstream/diarization/score.sh result/downstream/libri2mix_diar downstream/diarization/data/test
   ```

3. The scoring results will look like:

   ![](https://i.imgur.com/GnVlFlH.png)

   One should report the lowest number at the bottom, where the column represents DER and the most bottom row will always have the lowest DER which is the number we will report.

#### Re-check the scoring results

Running the above scoring script takes time. If you just want to re-check the scored results, use

```bash=
./downstream/diarization/report.sh result/downstream/libri2mix_diar
```

## ER: Emotion Recognition

#### Prepare data

1. Download dataset from https://sail.usc.edu/iemocap/

   - You will need to fill a form in IEMOCAP official website to get the dataset.

2. Preprocess by

```bash
python3 IEMOCAP_preprocess.py /path/to/IEMOCAP
```

3. Change the root path of IEMOCAP in `downstream/emotion/config.yaml`

#### Train

```bash
python3 run_downstream.py -m train -u fbank -d emotion -v fold1 -n emotion_lr1e-4_fold1
```

#### Test

The testing is done on-the-fly with training since it is not costly. Use the following command to get the testing result from the best-dev ckpt.

```bash
python3 utility/get_best_dev.py result/downstream/emotion_lr1e-4_fold1/log.log
```

#### Cross validation

5-fold cross validation is required for the standard evaluation of this dataset. The final ACC is averaged over 5 folds. If `-v` option is not assigned durining training it is default to **fold 1**. Available options for `-v` on this task are: **fold 1**, **fold 2**, **fold 3**, **fold 4**, **fold 5**. So you will need to:

```bash
for test_fold in fold1 fold2 fold3 fold4 fold5;
do
    python3 run_downstream.py -m train -u fbank -d emotion \
        -v $test_fold -n emotion_lr1e-4_$test_fold
    python3 utility/get_best_dev.py \
        result/downstream/emotion_lr1e-4_$test_fold/log.log
done
```

# Add new downstream tasks

Please first have a quick look at the [General Usage](#general-usage) to have a sense on the basic command pattern.

Each downstream task is defined by a **self-contained** folder under this [downstream](./) folder, like the task ASR is defined in [downstream/asr](./asr). Once a new folder is placed under this [downstream](./) folder, says `downstream/blabla/`, you can specify to run this new downstream task with `-d blabla` option in [run_downstream.py](../run_downstream.py) script.

By **self-contained** we mean there should be all the downstream specific materials under your task folder, including the definition for dataset, datalader,model, and loss. How to define these materials are completely free, while the only requirement is to provide an `expert.py` file with an `DownstreamExpert` **nn.module** at the root of your downstream folder, where 3 object methods are implemented: `get_dataloader`, `forward`, and `log_records`.

The fastest way to know how the framework works is to run a minimum example, so we provide a pseudo task [downstream/example/](./example/), which can always be ran up by:

1. [**Clone the repository and install packages**](../README.md#installation)
2. Run the pseudo task to get a feeling of the framework

   ```bash
   python3 run_downstream.py -u fbank -d example -n HelloWorld
   ```

Hence, you can refer to [downstream/example/expert.py](./example/expert.py) for the minimum requirement and implementation specification. Also, you can use [downstream/example/](./example/) as an initial template for developing a new downstream task.

**Note**. Please use _relative import_ in your downstream folder, in case we might want to rename or move the location for the `downstream` folder in future.
