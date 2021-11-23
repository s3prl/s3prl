# Downstream tasks

### Introduction
Self-supervised (SSL) pretrained models are not able to justify their effectiveness through pretraining loss. One always has to evaluate their performance with downstream tasks. Hence, it is crucial to collect a wide range of downstream tasks and make the evaluation pipeline as easy as possible to speed up the development cycle.

We develop several downstream tasks for evaluating SSL models, each of them is defined by a sub-folder under this **downstream** folder. We further select representative ones to form the following **benchmark**:
- [**SUPERB:** **S**peech processing **U**niversal **PER**formance **B**enchmark (paper)](https://arxiv.org/abs/2105.01051)

### How to use

#### I. General requirement

1. [Clone the repository and install dependencies](../../README.md#installation)
2. See the [General Usage](#general-usage) to have a sense on the conceptual usage

#### II A. Run the developed tasks

3. **Optional:** [**Register your customized pretrained model**](../upstream/example/README.md)

    - You can also start from evaluating [pretrained models available in this toolkit](../upstream/README.md#how-to-use).

4. Follow the task-specific usages

    - [**SUPERB Benchmark & Challenge**](./docs/superb.md)
    - [Tasks used in Mockingjay & TERA](./docs/mockingjay_tera.md)
    - [More tasks](./docs/more_tasks.md)

#### II B. Develop new tasks

3. Check [Add new downstream tasks](#add-new-downstream-tasks). Pull requests are highly welcome. Thanks!

# General usage
All of the downstream task follow the following command pattern, with a few task-specific adjustments which are detailed in the follow-up task-specific sections.

## Start a new downstream training experiment

```bash
cd s3prl/

# General pattern
python3 run_downstream.py -m train -n ExpName -u UpstreamName -d DownstreamName

# A directly runnable example without data preparation
python3 run_downstream.py -m train -n ExpName -u fbank -d example

# Finetune the upstream.
# Please use the last hidden_state. See issue #192
python3 run_downstream.py -m train -n ExpName -u fbank -d example -f -l -1

# An example with downloading / uploading models via the Hugging Face Hub.
# Use the credentials associated with your account on huggingface.co
HF_USERNAME=username HF_PASSWORD=password python3 run_downstream.py -m train -n ExpName -u UpstreamName -d DownstreamName --hub huggingface --push_to_hf_hub True
```

- `-m` or `--mode` specifies the **train/evaluate** mode
- `-u` or `--upstream` specifies the upstream pretrained model.
    - The available upstream can be checked by `-h`
    - Some examples: `-u fbank`, `-u tera`, `-u wav2vec2`, `-u hubert`
- `-d` or `--downstream` specifies the downstream task.
    - The available downstream can be checked by `-h`
    - Some examples: `-d asr`, `-d emotion`, `-d speech_commands`
    - Each available downstream task has its corresponding folder under `downstream/`. Eg. `-d asr` means we are using the task defined in `downstream/asr/`
    - `example` is a pseudo downstream task which is useful for testing the upstream model or as an initial template for developing a new downstream task
- Feature selection from an upstream: the output of an upstream is a dictionary, where each key's corresponding value is a list of Tensors all in `(batch_size, max_sequence_length_of_batch, hidden_size)`. The final selected feature for the downstream training depends on `-s` and `-l`. If it is a list of Tensors, we train a learnable weighted-sum (WS) on them.
    - `-s` or `--upstream_feature_selection` (str, default: "hidden_states"): **select a key** from the upstream output dict. There are at least one key supported: `hidden_states`. Its value is a list of Tensors in the layer order where `value[0]` is closed to the upstream input and `value[-1]` is closed to the upstream output.
    - `-l` or `--upstream_layer_selection` (int, default: None) if not specified, then the dict value selected by `-s` is the final selection. If specified, then select a specific index from the dict value selected by `-s`
    - Examples:
        - Select all layers of hidden states (WS): `-s hidden_states`
        - Select the first layer: `-s hidden_states -l 0`
        - Select the last layer: `-s hidden_states -l -1`
        - Select a middle layer: `-s hidden_states -l 2`
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
- `--hub` specifies the model Hub (PyTorch or Hugging Face) to retrieve the upstream model from. Default: `torch`

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

We wrap the model with **DistributedDataParallel** (DDP). By inserting `-m torch.distributed.launch --nproc_per_node {GPU_NUM}` between `python3` and `run_downstream.py`, you can directly turn the above **training** commands into distributed training. We support DDP for all the SUPERB tasks.

### When to use DDP

When you find the training is too slow **and** `config.runner.gradient_accumulate_steps` > 1, you can speed up the training by using multiple GPUs and decrease the steps for gradient accumulation. Note that the following settings are effectively the same:

1. `gradient_accumulate_steps`=4, 1 GPU
2. `gradient_accumulate_steps`=2, 2 GPUs
3. `gradient_accumulate_steps`=1, 4 GPUs

Please remember to adjust `gradient_accumulate_steps` when using different GPU number by override. Eg. `-o config.runner.gradient_accumulate_steps=2`

### How to use DDP

Says the single GPU is too slow when `gradient_accumulate_steps`=8, and you wish to speed it up with 4 GPUs.

#### First specify your GPU number
```bash
gpus=4;
distributed="-m torch.distributed.launch --nproc_per_node ${gpus}";
```

#### Simple training
```bash
python3 $distributed run_downstream.py -m train -n ExpName -u fbank -d example \
    -o config.runner.gradient_accumulate_steps=2
```

Note that currently PyTorch might not automatically terminate all spawned processes when this training terminates or crashes, and can lead to "Out of GPU memory" or "Address already used" error if you directly launch a new DDP training. Since hardware resources are not yet released properly in the previous run. You can use `pkill -f "your previous command"` to terminate all related processes.

#### Resume training

```bash
# The $distributed value should be same as the original training experiment.
# [ckpt] can be the path of a checkpoint or its residing directory.
python3 $distributed run_downstream.py -m train -e [ckpt]
```

#### Fault-tolerant training

```bash
./run_while.sh python3 $distributed run_downstream.py -m train -n ExpName \
    -u fbank -d example -a -o config.runner.gradient_accumulate_steps=2
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

## Running with Docker

We provide a Docker image that allows you to pull upstream models from the PyTorch or Hugging Face Hub, fine-tune on a downstream task, and push the training results (weights, configs, tensorboard traces etc) to the Hugging Face Hub.

In the root of the repo, first build the image with

```
docker build -t s3prl:latest .
```

Then run the container using the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) and with the data mounted as follows:

```
docker run --gpus all -it -P -v /path/to/superb/data:/app/data -e "upstream_model=model_name" -e "downstream_task=task_name" -e "HF_USERNAME=username" -e "HF_PASSWORD=passwd" s3prl
```

Here `model_name` and `task_name` correspond to one of the supported models / downstream tasks in `s3prl`, and `HF_USERNAME` and `HF_PASSWORD` are your account credentials for the [Hugging Face Hub](https://huggingface.co). By default, each task's `config.yaml` is used to set all the training parameters, but can be overridden with the `override` argument as follows:

```
docker run --gpus all -it -P -v /data/lewis/superb:/app/data -e "HF_USERNAME=username" -e "HF_PASSWORD=password" -e "override=config.optimizer.lr=1e-04" s3prl
```

# Add new downstream tasks

Each downstream task is defined by a **self-contained** folder under this [downstream](./) folder, like the task ASR is defined in [downstream/asr](./asr). Once a new folder is placed under this [downstream](./) folder, says `downstream/blabla/`, you can specify to run this new downstream task with `-d blabla` option in [run_downstream.py](../run_downstream.py) script.

By **self-contained** we mean there should be all the downstream specific materials under your task folder, including the definition for dataset, dataloader, model, and loss. How to define these materials are completely free, while the only requirement is to provide an `expert.py` file with an `DownstreamExpert` **nn.module** at the root of your downstream folder, where 3 object methods are implemented: `get_dataloader`, `forward`, and `log_records`.

The fastest way to know how the framework works is to run a minimum example, so we provide a pseudo task [downstream/example/](./example/), which can be always ran up by:

```bash
python3 run_downstream.py -u fbank -d example -n HelloWorld
```

Hence, you can refer to [downstream/example/expert.py](./example/expert.py) for the minimum requirement and implementation specification. Also, you can use [downstream/example/](./example/) as an initial template for developing a new downstream task.

#### Note 1

Please use _relative import_ in your downstream folder, in case we might want to rename or move the location for the `downstream` folder in the future.

#### Note 2

If you want to train your downstream task with distributed training, you should take care to use **DistributedSampler** when providing the training dataloader in your expert file.
