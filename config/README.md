# Config File Documentation

Description of parameters available in config file are listed here, checkout example configs for how to use.

## asr_model

### Optimizer

| Parameter     | Description                                                                                                                 | Note                                                               |
|---------------|-----------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| type          | The name of pytorch optimizer for training.                                                                                 | Tested: `Adam`/`Adadelta`                                          |
| learning_rate | Learning rate for optimizer.                                                                                                |                                                                    |
| joint_ctc     | Weight on CTC loss function for multi-task learning, see [Joint CTC training](https://arxiv.org/abs/1609.06773) for detail. Set it to `0` for attention-based seq2seq (e.g. LAS) learning with cross entropy only.| Pure CTC (set this parameter to `1.0`) is not available currently. |

### Encoder

| Parameter    | Description                                                                                                                                                    | Note                                   |
|--------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------|
| enc_type     | Encoder architecture. For `VGGBiRNN`, 4x input time subsampling will be performed with CNN.                                                                    | Available now:`VGGBiRNN`/`BiRNN`/`RNN` |
| sample_rate  | Sample rate for each RNN layer, concatenated with `_`. For each layer, the length of output on the time dimension will be input/`sample_rate`.                 |                                        |
| sample_style | The down sampling mechanism. `concat` will concatenate multiple time steps according to sample rate into one vector, `drop` will drop the unsampled timesteps. | Available now:`concat`/`drop`          |
| dim          | Number of cells for each RNN layer (per direction), concatenated with `_`.                                                                                     | Depth must match `sample_rate`         |
| dropout      | Dropout between each layer, concatenated with `_`.                                                                                                             | Depth must match `sample_rate`         |
| rnn_cell     | RNN cell of all layer.                                                                                                                                         | Tested: `LSTM`                         |

### Attention

| Parameter | Description                                                                                                                                      | Note                       |
|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------|
| att_mode  | Attention mechanism option. `dot` is the vanilla attention and `loc` indicates the [location-based attention](https://arxiv.org/abs/1506.07503). | Available now: `dot`/`loc` |
| dim       | The dimension of all networks in attention (if there are any).                                                                                   |                            |
| proj      | Apply an additional transform layer to encoder feature before calculating energy.                                                                |                            |
| num_head  | ``Under development...``                                                                                                                         | ``Under development...``   |


## clm

If enabled, adversarial training between ASR and CLM will be imposed. This would allow ASR to learn from additional text source. See the [paper](https://arxiv.org/abs/1811.00787) for more detailed description.

## solver


| Parameter             | Description                                                                                                                                                                                                | Note                                                                     |
|-----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| dataset               | Dataset name for experiment.                                                                                                                                                                               | Available now: `timit`/`librispeech`                                              |
| data_path             | File path of the pre-processed dataset.                                                                                                                                                                    |                                                                           |
| n_jobs                | Number of workers for [Pytorch Dataloader](https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader).                                                                    |                                                                           |
| max_timestep          |  Upper limit of the length of input, training data with acoustic feature longer than the limit will be dropped. (0 = No restriction)                                                                       | Only applies to training data.                                            |
| max_label_len         | Upper limit of the length of output, training data with label sequence longer than the limit will be dropped. (0 = No restriction)                                                                         | Only applies to training data.                                            |
| train_set             | List containing the splits to be used as training data. Check the name of csv files under `data_path`                                                                                                      | e.g. `['train-clean-100','train-clean-360']` when `dataset`=`librispeech` |
| batch_size            | Batch size for training.                                                                                                                                                                                   |                                                                           |
| apex                  | Enable faster optimization, see [APEX](https://github.com/NVIDIA/apex) for more details.                                                                                                                   | Install apex first before setting this option to `True`.                  |
| total_steps           | Total number of steps to train.                                                                                                                                                                            |                                                                           |
| tf_start              | Apply [Scheduled Sampling](https://arxiv.org/abs/1506.03099) with teacher forcing rate at the beginning of training. With teacher forcing rate set to 1, ASR will never sample input from its own output.  |                                                                           |
| tf_end                | Teacher forcing rate at the end of training (linearly decay during training, 1 for no scheduled sampling at all).                                                                                          |                                                                           |
| dev_set               | List containing the splits to be used as validation data.                                                                                                                                                  | See `train_set` for how to use.                                           |
| dev_batch_size        | Batch size for validation.                                                                                                                                                                                 |                                                                           |
| dev_step              | Interval between each validation step (i.e. validation will be performed every <dev_step> step).                                                                                                           |                                                                           |
| test_set              | List containing the splits to be decoded in the testing phase.                                                                                                                                             | See `train_set` for how to use.                                           |
| max_decode_step_ratio | The maximum decoding time step will be `max_decode_step_ratio`x`input length` during testing (in case beam search might never end). Decoding will stop if <eos> is predicted or max decoding step reached. |                                                                           |
| decode_beam_size      | Beam size for beam search algorithm.                                                                                                                                                                       |                                                                           |
| decode_ctc_weight     | The weight on CTC prefix probability during decoding, see [joint CTC decoding](https://arxiv.org/abs/1706.02737) for more detail.                                                                          |                                                                           |
| decode_lm_weight      | The weight on probability predicted by RNNLM during decoding, see [joint RNNLM decoding](https://arxiv.org/abs/1706.02737) for more detail.                                                                |                                                                           |
| decode_lm_path        | The path of pre-trained RNNLM.                                                                                                                                                                             |                                                                           |