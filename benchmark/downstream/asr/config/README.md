# Config File Documentation

Description of parameters available in config files for [training](##Training-Configs) and [inference](##Inference-Configs).

## Training Configs

Each config should include `data`/`hparas`/`model`, see [example on LibrSpeech](libri/asr_example.yaml).

### Data

Options under this category are all data-related.

- Corpus

    For each corpus, a corresponding source python file `<corpus_name>.py` file should be placed at `corpus/`, checkout [`librispeech.py`](../corpus/librispeech.py) for example.

    |Parameter | Description | Note |
    |----------|-------------|------|
    | name     | `str` name of corpus (used in [`data.py`](../src/data.py) to import the dataset defined in `<corpus_name>.py`) | Available: `Librispeech`|
    | path     | `str` path to the specified corpus, parsing file structure should be handled in `<corpus_name>.py` |  |
    | train_split| `list` which includes subsets of corpus used for training, accepted partition names should be defined in `<corpus_name>.py`||
    | dev_split | `list` which includes subsets of corpus used for validation, accepted partition names should be defined in `<corpus_name>.py`||
    | bucketing | `bool` to enable bucketing, i.e. similar length in each batch, should be implemented in `<corpus_name>.py`| More effecient training but biased sampling|
    | batch_size | `int` Batch size for training/validation, will be send to Torch Dataloader ||

- Audio

    Hyperparameters of feature extraction performed on-the-fly mostly done by [torchaudio](https://pytorch.org/audio/), checkout [audio.py](../src/audio.py) for implementation.
    
    |Parameter | Description | Note |
    |----------|-------------|------|
    | feat_type| `str` name of audio feature to be used. Please note that MFCC required latest torch audio | Available: `fbank`/`mfcc`|
    | feat_dim| `int` dimensionality of audio feature, if you are not fimiliar with audio features, `40` for `fbank` and `13` for `mfcc` generally works||
    | frame_length | `int` size of the window (millisecond) for feature extraction ||
    | frame_shift |  `int` hop size of the window (millisecond) for feature extraction ||
    | dither | `float` dither when extracting feature | See [doc](https://pytorch.org/audio/compliance.kaldi.html#functions)|
    | apply_cmvn | `bool` to activate feature normalization | Using our own implementation |
    | delta_order | `int` to apply delta on feature. <p> `0`: do nothing, `1`: add delta, `2`: also add accelerate | Using our own implementation|
    | delta_window_size | `int` to specify the window size for delta calculation ||

- Text

    Options to specify how text should be encoded, subword models use [sentencepiece](https://github.com/google/sentencepiece)
    
    |Parameter | Description | Note |
    |----------|-------------|------|
    | mode | `str` text unit for encoding sentences | Available: `character`/`subword`/`word`|
    | vocab_file | `src` path to file containing vocabulary set| Please use [generate_vocab_file.py](../util/generate_vocab_file.py) to generate it|

### Hparas

Options under this category are all training-related.

| Parameter     | Description | Note |
|---------------|-------------|------|
| valid_step    | `int` interval, numbers of training step for each validation |
| max_step      | `int` total training step |
| tf_start      | `float` init. teacher forcing probability in scheduled sampling | |
| tf_end        | `float` final teacher forcing probability in scheduled sampling | |
| tf_step       | `int` number of steps to linearly decrease teacher forcing probability|
| optimizer     | `str` the name of pytorch optimizer for training| Tested: `Adam`/`Adadelta`|
| lr            | `float` learning rate for optimizer |  |
| eps           | `float` epsilon for optimizer |  |
| lr_scheduler  | `str` learning rate scheduler | Available: `fixed`/`warmup`|
| curriculum    | `int` numbers of epochs to perform curriculum learning (short uttr. first) | |

### Model


- `ctc_weight`: weight of CTC in hybird CTC-Attention model (between `0~1`, `0`=disabled, `1` is under development)
- Encoder

    | Parameter    | Description  | Note |
    |--------------|--------------|------|
    | prenet       | `str` to employ VGG/CNN based encoder before RNN | [`vgg`](https://arxiv.org/pdf/1706.02737.pdf)/`cnn` |
    | module       | `str` the name of recurrent unit for encoder RNN layer | Only `LSTM` was tested |
    | bidirection  | `bool` to enable bidirectional RNN over input sequence | |
    | dim          | `list` of number of cells for each RNN layer (per direction)| |
    | dropout      | `list` of dropout probability for each RNN layer| Length must match `dim`  |
    | layer_norm   | `list` of `bool` to enable LayerNorm for each RNN layer | Not recommended |
    | proj   | `list` of `bool` to enable linear projection after each RNN layer | Length must match `dim`  |
    | sample_rate  | `list` sample rate for each RNN layer. For each layer, the length of output on the time dimension will be input/`sample_rate`.| Length must match `dim`          |
    | sample_style | `str` the down sampling mechanism. `concat` will concatenate multiple time steps according to sample rate into one vector, `drop` will drop the unsampled timesteps. | Available:`concat`/`drop`          |

- Attention

    | Parameter | Description | Note |
    |-----------|-------------|------|
    | mode  | `str` attention mechanism, `dot` is the vanilla attention and `loc` indicates the [location-based attention](https://arxiv.org/abs/1506.07503). | Available: `dot`/`loc` |
    | dim       | `int` dimension of all networks in attention |  |
    | num_head  | `int` number of head in [multi-head attention](https://arxiv.org/pdf/1706.03762.pdf), `1`: normal attention | Performance untested   |
    | v_proj    | `bool` to apply additional linear transform to encoder feature before weighted sum | |
    | temperature    | `float` the temperature to controll sharpness of sofmax function in attention | |
    | loc_kernel_size  | `int` kernel size for convolution in [location awared attention](https://arxiv.org/pdf/1506.07503.pdf) | For `loc` only |
    | loc_kernel_num  | `int` number of kernel for convolution in [location awared attention](https://arxiv.org/pdf/1506.07503.pdf) | For `loc` only |

- Decoder

    | Parameter    | Description  | Note |
    |--------------|--------------|------|
    | module       | `str` the name of recurrent unit for encoder RNN layer | Only `LSTM` was tested |
    | dim          | `int` number of cells in decoder| |
    | layer        | `int` number of layers in decoder | |
    | dropout      | `float` of dropout probability | |
  

### Additional Plug-ins

The following mechanisms are our proposed methods, can be activate by inserting these parameters to config file

- Emb

    | Parameter    | Description  | Note |
    |--------------|--------------|------|
    | enable       | `bool` to enable word embedding regularization or fused decoding on ASR | |
    | src          | `str` path to pre-trained embedding table or BERT model| The `bert-base-uncased` model fine-tuned on librispeech text data is available [here](https://drive.google.com/file/d/1Y1q5cH3yfuzMxQArR7WJ4gQD1GN7xrPh/view?usp=sharing) |
    | distance     | `str` measurement of distance between word embedding and model output | Available: `CosEmb`/`MSE`(untested) |
    | weight       | `float` $\lambda$ in paper  | |
    | fuse         | `float` $\lambda_f$ in paper| |
    | fuse_normalize| `bool` to normalize output before Cosine-Softmax in paper, should be on when `distance==CosEmb` | |
    | bert         | `str` name of BERT model if using BERT as target embedding, e.g. `bert-base-uncased`| mutually exclusive to `fuse>0`|



## Inference Configs

Each config should include `src`/`decode`/`data`, see [example on LibrSpeech](libri/decode_example.yaml).
Note that most of the options (audio feature, model structure, etc.) will be imported from the training config specified in `src`.

### Src

Specify the ASR to use in decoding process.

| Parameter | Description  | Note |
|-----------|--------------|------|
| ckpt      | `str` path to ASR checkpoint to be load | |
| config    | `str` path to ASR training config which belongs to the checkpoint| |

### Data

- Corpus

    |Parameter | Description | Note |
    |----------|-------------|------|
    | name     | See `corpus` section in training config||
    | dev_split| See `corpus` section in training config||
    | test_split| Like dev set, ASR will perform exactly same decoding process on this set, should also be defined by user like train/dev set||


### Decode

Options for decoding that *will* dramatically change the decoding result.

| Parameter | Description  | Note |
|-----------|--------------|------|
| beam_size | `int` beam size for beam search algorithm, be careful that larger beam increases memory usage||
| min_len_ratio | `float` the minimum length of any hypothesis will be `min_len_ratio` x `input length` |
| max_len_ratio | `float` the maximum decoding time step will be `max_len_ratio` x `input length`, hypothesis will end if `<eos>` is predicted or maximum decoding step reached |
| lm_path   | `str` the path to pre-trained LM for joint decoding, **this is not language model rescoring**| [paper](https://arxiv.org/pdf/1706.02737.pdf)|
| lm_config | `str` the path to the config of pre-trained LM for joint decoding| [paper](https://arxiv.org/pdf/1706.02737.pdf) |
| lm_weight | `float` the weight for RNNLM in joint decoding| [paper](https://arxiv.org/pdf/1706.02737.pdf), slower inference |
| ctc_weight| `float` the weight for CTC network in joint decoding, this will only be available if `ctc_weight` was not zero in training config | [paper](https://arxiv.org/pdf/1706.02737.pdf), slower inference |

