# Pre-training  Upstream Models
For pre-training new upstream models, you'll need high-end GPU(s).

## Step 1) Prepare data
1) Download the LibriSpeech raw data from [here](http://www.openslr.org/12).
    - These sets are used for pretraining:
        - train-clean-100 [6.3G]
        - train-clean-360 [23G]
        - train-other-500 [30G]
    - The LibriSpeech directory after download and unzip should look like this: 
      ![](https://i.imgur.com/PdAOXjq.png)
2) **(IMPORTANT)** Generate the meta data directory `len_for_bucket/` for bucketing to accelerate training: 
```bash
python3 preprocess/generate_len_for_bucket.py -i PATH_TO_YOUR/LibriSpeech/
```

## Step 2) Modifiy runner config
1) Open `S3PRL/pretrain/tera/config_runner.yaml`:
    - This is the default runner config of tera, and it will be taken by default if not specified otherwise.
    - To assign another config other then default, you can use the `-c` argument, for example:
      `-u tera -c another_new_config_runner.yaml`
      
2) Change the paths in `config_runner.yaml`:
    - Change the following to your own path:
    ```yaml
    libri_root: '/media/andi611/1TBSSD/LibriSpeech/'
    file_path: 'data/len_for_bucket' 
    ```
3) Other training settings to care about:
    - Check these attributes in `config_runner.yaml`:
    ```yaml
    n_epochs: 100
    total_steps: -1
    gradient_accumulate_steps: 8
    train_batch_size: 32
    sets: ['train-clean-100', 'train-clean-360', 'train-other-500']
    ```
    - If `n_epochs` is given, `total_steps` will be ignored.
    - Set `n_epochs` to `-1` if you want to use `total_steps` instead
    - The acutal batch size = `gradient_accumulate_steps` * `train_batch_size`
    - Modify the `sets` list to choose pretraining subsets.

    
## Step 3) Start training

### Mockingjay
- Command:
```bash
python run_pretrain.py -u mockingjay -g pretrain/mockingjay/config_model.yaml -n YourModelName
```
- This takes the default runner conifg:`pretrain/mockingjay/config_runner.yaml`
- Use `-c` to specify runner config other than default.
- On multiple GPUs, add `--multi_gpu`
- Check the log on console that everything is correct and as expected:
  ![](https://i.imgur.com/mbmtGOH.png)

### TERA
- Command:
```bash
python run_pretrain.py -u tera -g pretrain/tera/config_model.yaml -n YourModelName
```
- Everything else is identical to the above.

### Audio ALBERT
- Command:
```bash
python run_pretrain.py -u audio_albert -g pretrain/audio_albert/config_model.yaml -n YourModelName
```
- Everything else is identical to the above.

### Distiller
- Command:
```bash
python run_pretrain.py -u distiller -g pretrain/distiller/config_model.yaml -n YourModelName
```
- Everything else is identical to the above.

## Step 4) Loading the pre-trained checkpoint for downstream tasks
### Mockingjay
- Use the local method of `mockingjay_local` and `-k` to specify the path to your checkpoint.
- Example command:
```
python run_downstream.py -m train -u mockingjay_local -k example_checkpoint/states-1000000.ckpt -d phone_linear -n ExpName
```

### TERA
- Use the local method of `tera_local` and `-k` to specify the path to your checkpoint.
- Example command:
```bash
python run_downstream.py -m train -u tera_local -k example_checkpoint/states-1000000.ckpt -d phone_linear -n ExpName
```

### AUDIO ALBERT
- Use the local method of `audio_albert_local` and `-k` to specify the path to your checkpoint.
- Example command:
```bash
python run_downstream.py -m train -u audio_albert_local -k example_checkpoint/states-1000000.ckpt -d phone_linear -n ExpName
```
