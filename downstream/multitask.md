# General usage
All of the downstream task follow the following command pattern, with a few task-specific adjustments which are detailed in the follow-up task-specific sections.

## Start a new downstream training experiment

```bash
# general pattern
python3 run_downstream.py -m train -n ExpName -u UpstreamName -d Downstream1,Downstream2,Downstream3
```

- `-m` or `--mode` specifies the **train/evaluate** mode
- `-c` or `--config` specifies the config file path. Must be specified and see [multitask.yaml](./multitask.yaml) for an example.
- `-u` or `--upstream` specifies the upstream pretrained model.
    - The available upstream can be checked by `-h`
- `-d` or `--downstream` specifies the downstream tasks defined in the config file you want to consider in this training. If you specify `-d asr,sid -c downstream/multitask.yaml` then only two tasks in [multitask.yaml](./multitask.yaml) will be used.
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

The following `test-clean` is an example for the **name** of the testing dataset of ASR; `test` is the **name** of the testing dataset of SID. The supported **name** is defined by each downstream expert's `get_dataloader`. Typically `dev` and `test` are supported for task/dataset with the standard split.

### Preferable: Use the same args & config as training time

```bash
# [ckpt] can be the path of a checkpoint or its residing directory.
python3 run_downstream.py -m evaluate -t "asr:test-clean,sid:test" -e [ckpt]
```

- The `-e` or `--past_exp` option is designed to use the exact same arguments and config as the previous training experiment **except the training/evaluation mode** and the **testing split**. (Each checkpoint will save arguments and config.)
- `-o` can be used to further override the arguments & configs loaded from the checkpoint, since `-o` is at the highest priority.

### Test the distributed trained checkpoint

Only the training part is powered by **DistributedDataParallel**, and we save all the model *state_dict* **without** the DDP wrapper. That is, after the DDP training, you can always evaluate the checkpoint using the testing command documented above (on single GPU).