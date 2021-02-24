# Upstream Documentation

Table of Contents
------------------------------------

<!--ts-->
   * [Table of contents](#table-of-contents)
   * [1-for-all interface](#1-for-all-interface)
   * [Upstream Self-Supervised Models](#upstream-self-supervised-models)
      * [How to specify your upstream model](#how-to-specify-your-upstream-model)
      * [Knowing your upstream model](#knowing-your-upstream-model)
      * [Use upstream models in your own project](#use-upstream-models-in-your-own-project)
   * [Upstream Acoustic Feature Extracters](#upstream-acoustic-feature-extracters)
      * [Knowing your feature extracter](#knowing-your-feature-extracter)
      * [Use feature extracters in your own project](#use-feature-extracters-in-your-own-project)
<!--te-->

------------------------------------

## 1-for-all interface
We provide an all-in-one unified interface for numerous speech pretrained models.
**All the upstream models take input / output of the same format:**
- **input**: list of unpadded wavs `[wav1, wav2, ...]`, each wav is in `torch.FloatTensor`
- **output**: list of unpadded representations `[rep1, rep2, ...]`, each erp is of the shape `(extracted_seqlen, feature_dim)`

For upstream models that operate on features other than wav (for example: log Mel, fbank, etc), the preprocessing of wav -> feature is done on-they-fly during model forward. Rest assured that this will not increase your runtime.

## Upstream Self-Supervised Models
Below is a list of available upstream models that we currently support: 

Publication Date | Model | name | Paper | Input | Stride | Pre-train | Ckpt | Repo 
|---|---|---|---|---|---|---|---|---
10 Jul 2018 | CPC | cpc | [arxiv](https://arxiv.org/abs/1807.03748) | wav | 10ms | [LibriLight-60k](https://github.com/facebookresearch/libri-light) | X | [FAIR](https://github.com/facebookresearch/CPC_audio)
5 Apr 2019 | APC | apc | [arxiv](https://arxiv.org/abs/1904.03240) | Mel | 10ms | [LibriSpeech-360](http://www.openslr.org/12) | O | [APC](https://github.com/iamyuanchung/Autoregressive-Predictive-Coding)
6 Apr 2019 | PASE | pase | [arxiv](https://arxiv.org/abs/1904.03416) | wav | 10ms | [LibriSpeech-960](http://www.openslr.org/12) | X | [PASE](https://github.com/santi-pdp/pase)
11 Apr 2019 | Wav2Vec | wav2vec | [arxiv](https://arxiv.org/abs/1904.05862) | wav | 10ms | [LibriSpeech-960](http://www.openslr.org/12) | O | [Fairseq](https://github.com/pytorch/fairseq)
12 Oct 2019 | VQ-Wav2Vec | vq_wav2vec | [arxiv](https://arxiv.org/abs/1910.05453) | wav | 10ms | [LibriSpeech-960](http://www.openslr.org/12) | O | [Fairseq](https://github.com/pytorch/fairseq)
25 Oct 2019 | Mockingjay | mockingjay | [arxiv](https://arxiv.org/abs/1910.12638) | Mel | 10ms | [LibriSpeech-960](http://www.openslr.org/12) | O | [S3PRL](https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning)
7 Feb 2020 | Modified-CPC | cpc | [arxiv](https://arxiv.org/abs/2002.02848) | wav | 10ms | [LibriLight-60k](https://github.com/facebookresearch/libri-light) | O | [FAIR](https://github.com/facebookresearch/CPC_audio)
17 May 2020 | VQ-APC | vq_apc | [arxiv](https://arxiv.org/abs/2005.08392) | Mel | 10ms | [LibriSpeech-360](http://www.openslr.org/12) | O | [NPC](https://github.com/iamyuanchung/Autoregressive-Predictive-Coding)
18 May 2020 | Audio Albert | audio_albert | [arxiv](https://arxiv.org/abs/2005.08575) | Mel | 10ms | [LibriSpeech-960](http://www.openslr.org/12) | X | [S3PRL](https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning)
20 Jun 2020 | Wav2Vec 2.0 | wav2vec2 | [arxiv](https://arxiv.org/abs/2006.11477) | wav | 20ms | [LibriSpeech-960](http://www.openslr.org/12) | O | [Fairseq](https://github.com/pytorch/fairseq)
12 Jul 2020 | TERA | tera | [arxiv](https://arxiv.org/abs/2007.06028) | Mel | 10ms | [LibriSpeech-960](http://www.openslr.org/12) | O | [S3PRL](https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning)
1 Nov 2020 | NPC | npc | [arxiv](https://arxiv.org/abs/2011.00406) | Mel | 10ms | [LibriSpeech-360](http://www.openslr.org/12) | O | [NPC](https://github.com/iamyuanchung/Autoregressive-Predictive-Coding)

<details><summary>Upstreams ordered in different style of losses</summary><p>

* **Mockingjay**
    - Described in ["Mockingjay: Unsupervised Speech Representation Learning with Deep Bidirectional Transformer Encoders"](https://arxiv.org/abs/1910.12638)
    - *Transformer based, BERT-style masked reconstruction loss*
    - These papers used our implementations: [Adversarial Defense](https://arxiv.org/abs/2006.03214), [Understanding Self-attention](https://arxiv.org/abs/2006.03265)
    - Checkpoints are provided by this repo: [s3prl](https://github.com/s3prl/s3prl)
* **TERA**
    - Described in ["TERA: Self-Supervised Learning of Transformer Encoder Representation for Speech"](https://arxiv.org/abs/2007.06028)
    - *Transformer based, Advanced masked reconstruction loss*
    - Checkpoints are provided by this repo: [s3prl](https://github.com/s3prl/s3prl)
* **Audio ALBERT**
    - Described in ["Audio ALBERT: A Lite BERT for Self-supervised Learning of Audio Representation"](https://arxiv.org/abs/2005.08575)
    - *Transformer based, BERT-style masked reconstruction loss*
    - Checkpoints are provided by this repo: [s3prl](https://github.com/s3prl/s3prl)
* **CPC**
    - Described in ["Representation Learning with Contrastive Predictive Coding"](https://arxiv.org/abs/1807.03748)
    - *CNN based, InfoNCE contrastive loss*
    - Checkpoints are provided by this repo: [FAIR](https://github.com/facebookresearch/CPC_audio)
* **APC**
    - Described in ["An Unsupervised Autoregressive Model for Speech Representation Learning"](https://arxiv.org/abs/1904.03240)
    - *RNN based, unidirectional reconstruction loss*
    - Checkpoints are provided by this repo: [Alexander-H-Liu/NPC](https://github.com/Alexander-H-Liu/NPC)
* **VQ-APC**
    - Described in ["Vector-Quantized Autoregressive Predictive Coding"](https://arxiv.org/abs/2005.08392)
    - *RNN based, unidirectional reconstruction loss + vector quantization*
    - Checkpoints are provided by this repo: [Alexander-H-Liu/NPC](https://github.com/Alexander-H-Liu/NPC)
* **NPC**
    - Described in ["Non-Autoregressive Predictive Coding for Learning Speech Representations from Local Dependencies"](https://arxiv.org/abs/2011.00406)
    - *CNN based, reconstruction loss with Masked Convolution Blocks*
    - Checkpoints are provided by this repo: [Alexander-H-Liu/NPC](https://github.com/Alexander-H-Liu/NPC)
* **wav2vec**
    - Described in ["wav2vec: Unsupervised Pre-training for Speech Recognition"](https://arxiv.org/abs/1904.05862)
    - *CNN based, InfoNCE contrastive loss*
    - Checkpoints are provided by this repo: [fairseq](https://github.com/pytorch/fairseq/blob/master/examples/wav2vec/README.md)
* **vq-wav2vec**
    - Described in ["vq-wav2vec: Self-Supervised Learning of Discrete Speech Representations"](https://arxiv.org/abs/1910.05453)
    - *CNN based, InfoNCE contrastive loss*
    - Checkpoints are provided by this repo: [fairseq](https://github.com/pytorch/fairseq/blob/master/examples/wav2vec/README.md)
* **wav2vec 2.0**
    - Described in ["wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations"](https://arxiv.org/abs/2006.11477)
    - *CNN+Transformer based, InfoNCE contrastive loss + vector quantization + BERT-style masking*
    - Checkpoints are provided by this repo: [fairseq](https://github.com/pytorch/fairseq/blob/master/examples/wav2vec/README.md)
</p></details>

[Back to Top](#table-of-contents)

------------------------------------

### How to specify your upstream model
The `Name` field is the tag we used in this repo to specify different models. In other words, different upstream are identified with the exact string of their `Name`. For example, this is how you call an upstream through the hub:
```python
model_1 = torch.hub.load('s3prl/s3prl', 'cpc') # build the CPC model with pre-trained weights
model_2 = torch.hub.load('s3prl/s3prl', 'tera') # build the TERA model with pre-trained weights
model_3 = torch.hub.load('s3prl/s3prl', 'wav2vec2') # build the Wav2Vec 2.0 model with pre-trained weights
```
For example, this is how you specifing them in the command of downstream training scripts (more details of downstream tasks can be found [here](https://github.com/s3prl/s3prl/tree/master/downstream#downstream-tasks):
```bash
python run_downstream.py -m train -u npc -d example -n NameOfExp # Using the NPC model with pre-trained weights on downstream tasks
```
You can get a list of all available upstream `name`s through the following:
```python
import torch
print(torch.hub.list('s3prl/s3prl'))
```

[Back to Top](#table-of-contents)

------------------------------------

### Knowing your upstream model
* **Input**: The `Input` field shows which type of input the upstream model takes. *Note that although there are a varity of input types, but our interface always you to always feed waveform  as input* (`list of unpadded wavs [wav1, wav2, ...]`). For upstream models that use input other than wav (e.g. Mel), the preprocessing process is done on-the-fly during model forward. Rest assured that this will not increase your runtime.
* **Pre-train**: The `Pre-train` field tells on which data the upstream model was pre-trained, so you can make acceptable plug-in of certain model into certain pipeline.
* **Stride**: The `Stride` field shows the timeframe that each extracted feature encodes. In other words, the upstream model extracts a representation every `Stride` (ms).

[Back to Top](#table-of-contents)

------------------------------------

### Use upstream models in your own project
* Models with pretrained weights are provided with a convenient [torch.hub](https://pytorch.org/docs/stable/hub.html) interface.
Use `torch.hub.load('s3prl/s3prl', Name)` in your python scripts to build a pre-trained upstream model. Here is a simple example:
```python
import torch
device = 'cuda' # or cpu
upstream = torch.hub.load('s3prl/s3prl', 'tera').to(device)
wavs = [torch.zeros(160000, dtype=torch.float).to(device) for _ in range(16)] # list of unpadded wavs `[wav1, wav2, ...]`, each wav is in `torch.FloatTensor`
with torch.no_grad():
    reps = upstream(wavs) # list of unpadded representations `[rep1, rep2, ...]`, each erp is of the shape `(extracted_seqlen, feature_dim)`
```

* <details><summary>Click here to see an example code with argument parser</summary><p>

    ```python
    import torch
    import argparse

    parser = argparse.ArgumentParser(description='Minimum arguments for the pretrained upstream')
    parser.add_argument(
        '--upstream',
        help='Specify the upstream variant according to torch.hub.list'
    )
    parser.add_argument(
        '--upstream_feature_selection',
        help=f'Specify the layer to be extracted as the representation according to torch.hub.help'
    )
    parser.add_argument(
        '--upstream_refresh',
        action='store_true',
        help='Re-download cached ckpts for on-the-fly upstream variants'
    )
    parser.add_argument(
        '--upstream_ckpt',
        metavar='{PATH,URL,GOOGLE_DRIVE_ID}',
        help='Only set when the specified upstream has \'ckpt\' as an argument in torch.hub.help'
    )
    parser.add_argument(
        '--upstream_trainable', action='store_true',
        help='To fine-tune the whole upstream model'
    )
    args = parser.parse_args()

    device = 'cuda' # or cpu
    upstream = torch.hub.load(
        's3prl/s3prl',
        args.upstream,
        feature_selection = args.upstream_feature_selection,
        refresh = args.upstream_refresh,
        ckpt = args.upstream_ckpt,
    ).to(device)
    ```
</p></details>

* There are also example codes under each upstream directory (`upstream/*/example_extract.py`) for your reference. 
* In addition, here is a **detailed tutorial: [slides](https://docs.google.com/presentation/d/1n2Twz8YEmX67k6Vs_9aIzR6arVacnWzEFZNxBl-jsKU/edit?usp=sharing)**

[Back to Top](#table-of-contents)

------------------------------------

## Upstream Acoustic Feature Extracters
| Feature | Name | Default Dim | Hop | Window | Backend |
| -------- | -------- | -------- | -------- | -------- | -------- |
| Spectrogram | spectrogram | 257 | 10ms | 25ms | torchaudio-kaldi |
| FBANK | fbank | 80 + delta1 + delta2 | 10ms | 25ms | torchaudio-kaldi |
| MFCC | mfcc | 13 + delta1 + delta2 | 10ms | 25ms | torchaudio-kaldi |
| Mel | mel | 80 | 10ms | 25ms | torchaudio |
| Linear | linear | 201 | 10ms | 25ms | torchaudio |

[Back to Top](#table-of-contents)

------------------------------------

### Knowing your feature extracter
* **Name**: The `Name` field is the tag we used in this repo to specify different acoustic features. In other words, different features are identified with the exact string of their `Name`. For example, this is how you call an feature extracter through the hub:
```python
extracter_1 = torch.hub.load('s3prl/s3prl', 'fbank') # build the FBANK extractor with default config
extracter_2 = torch.hub.load('s3prl/s3prl', 'mel') # build the Mel extractor with default config
```
For example, this is how you specifing them in the command of downstream training scripts (more details of downstream tasks can be found [here](https://github.com/s3prl/s3prl/tree/master/downstream#downstream-tasks):
```bash
python run_downstream.py -m train -u mfcc -d example -n NameOfExp # Using the MFCC extracter with default configs on downstream tasks
```

* **Default Dim**: The `default dim` field tells the number of dimension a particular feature has by default. The dim can be changed by modifying the config files, they can be found at `s3prl/upstream/baseline/*.yaml`. Feel free to change them to fit your situation.

* **Backend**: The `backend` field tells how the feature extraction is based on. The `torchaudio-kaldi` backend matches the input/output of Kaldi’s compute-spectrogram-feats, see [torchaudio.compliance.kaldi](https://pytorch.org/audio/stable/compliance.kaldi.html) for more information. The `torchaudio` backend uses standard preprocessing procedures, see [torchaudio.transforms](https://pytorch.org/audio/stable/transforms.html) for more information.

[Back to Top](#table-of-contents)

------------------------------------

### Use feature extracters in your own project
Feature extracters with default config are provided with a convenient [torch.hub](https://pytorch.org/docs/stable/hub.html) interface.
Use `torch.hub.load('s3prl/s3prl', Name)` in your python scripts to build a feature extracter with default config. Here is a simple example:
```python
import torch
device = 'cuda' # or cpu
extracter = torch.hub.load('s3prl/s3prl', 'mfcc').to(device)
wavs = [torch.zeros(160000, dtype=torch.float).to(device) for _ in range(16)]
with torch.no_grad():
    mfcc = extracter(wavs)
```

To use with a modified config:
```python
import torch
device = 'cuda' # or cpu
extracter = torch.hub.load('s3prl/s3prl',
                           'baseline_local',
                            config='upstream/baseline/mfcc.yaml').to(device)
wavs = [torch.zeros(160000, dtype=torch.float).to(device) for _ in range(16)]
with torch.no_grad():
    mfcc = extracter(wavs)
```

[Back to Top](#table-of-contents)

------------------------------------
