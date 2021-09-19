# SUPERB Artifacts

## Prerequisite

Please read [downstream/README.md](../README.md) for the general command pattern, and [downstream/docs/superb.md](./superb.md) for the task-specific usage.

## Introduction

[**Released Artifacts**: http://140.112.21.28:8000/](http://140.112.21.28:8000/)

We release the Tensorboard logs and the trained downstream weights (the best on public dev set) for you to quickly understand the performance trend of a new upstream model. You can use `wget -r` to download the folder you want. e.g.

```bash
wget -r http://140.112.21.28:8000/pr/
```

To inference with the trained weights, you need to first prepare the dataset for each task following the above **task-specific usage**. Then, follow the task-specific commands below. There is an easy & general pattern:

**1. Override the dataset path**

Since the released checkpoints contain the dataset paths at our side, which is not valid in your local workspace. Please use `-o config.downstream_expert...` override function to change them.

**2. Override the expdir**

Since the released checkpoint contains the `expdir` specified during our experiment, which will be the saving directory for the inferenced prediction files but will also be hard to find for you. Please use `-o args.expdir=` override function to change it.

## Task-specific commands

### PR

```bash
data="your librispeech root"

python3 run_downstream.py -m evaluate -t test \
    -e checkpoint.ckpt \
    -o args.expdir=tgt_dir/pr,,config.downstream_expert.corpus.path=$data
```

### ASR

```bash
data="your librispeech root"

python3 run_downstream.py -m evaluate -t test-clean \
    -e checkpoint.ckpt \
    -o args.expdir=tgt_dir/asr,,config.downstream_expert.datarc.libri_root=$data
```

### KS

```bash
train="your speech command train root"
test="your speech command test root"

python3 run_downstream.py -m evaluate -t test \
    -e checkpoint.ckpt \
    -o args.expdir=tgt_dir/ks,,config.downstream_expert.datarc.speech_commands_root=$train,,config.downstream_expert.datarc.speech_commands_test_root=$test
```

### QbE

QbE uses Dynamic Time Warping which does not involve trainable parameters. We release the best layer (indexed from 0) we found on dev set.

| Upstream   | wav2vec 2.0 Base | HuBERT Base | wav2vec 2.0 Large | HuBERT Large |
| ---------- | ---------------- | ----------- | ----------------- | ------------ |
| Best layer |         1        |     12      |        15         |      24      |

By following the **task-specific** usage you can easily reproduce the results on QbE.

### SID

```bash
data="your voxceleb1 path"

python3 run_downstream.py -m evaluate -t test \
    -e checkpoint.ckpt \
    -o args.expdir=tgt_dir/sid,,config.downstream_expert.datarc.file_path=$data
```

### ASV

```bash
data="your voxceleb1 path"

python3 run_downstream.py -m evaluate -t test \
    -e checkpoint.ckpt \
    -o args.expdir=tgt_dir/asv,,config.downstream_expert.datarc.file_path=$data
```

### SD

If you follow the kaldi-style data preparation in the **task-specific usage**, then you don't need to override data paths for this task.

```bash
python3 run_downstream.py -m evaluate -t test \
    -e checkpoint.ckpt \
    -o args.expdir=tgt_dir/sd
```

### IC

```bash
data="your fluent speech root"

python3 run_downstream.py -m evaluate -t test \
    -e checkpoint.ckpt \
    -o args.expdir=tgt_dir/ic,,config.downstream_expert.datarc.file_path=$data
```

### SF

```bash
data="your audio snips root"

python3 run_downstream.py -m evaluate -t test \
    -e checkpoint.ckpt \
    -o args.expdir=tgt_dir/pr,,config.downstream_expert.corpus.path=$data,,config.downstream_expert.text.slots_file=$data/slots.txt
```

### ER

```bash
data="your IEMOCAP root"

python3 run_downstream.py -m evaluate -t test \
    -e checkpoint_fold1.ckpt \
    -o args.expdir=tgt_dir/er/fold1,,config.downstream_expert.datarc.root=$data
```

### SS

```bash
Will be released soon.
```

### SE

```bash
Will be released soon.
```

### ST

```bash
Will be released soon.
```
