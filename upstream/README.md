# Upstream Models

## 1-for-all interface
We provide an all-in-one unified interface for numerous speech pretrained models.
All the models take input / output of the same format:
- **input**: list of unpadded wavs `[wav1, wav2, ...]`, each wav is in torch.FloatTensor
- **output**: list of unpadded representations `[rep1, rep2, ...]`, each erp is of the shape: (extracted_seqlen, feature_dim)

## Easy Setup
Models with pretrained weights are provided with a convenient [torch.hub](https://pytorch.org/docs/stable/hub.html) interface.
Use `torch.hub.load('s3prl/s3prl', MODEL_NAME)` in your python scripts to build a pre-trained upstream model.
Here is a simple example:
```python
import torch
model = torch.hub.load('s3prl/s3prl', 'tera')
wavs = [torch.zeros(160000, dtype=torch.float) for _ in range(16)]
repr = model(wavs)
```
Check [here](https://docs.google.com/presentation/d/1n2Twz8YEmX67k6Vs_9aIzR6arVacnWzEFZNxBl-jsKU/edit?usp=sharing) for a detailed tutorial.

## Available Upstream Models
Below is a list of available upstream models that we currently support. The `name` field is the tag we used in this repo to specify different models. In other words, different upstream are identified with the exact string of their `name`, for example:
```python
model_1 = torch.hub.load('s3prl/s3prl', 'cpc')
model_2 = torch.hub.load('s3prl/s3prl', 'tera')
model_3 = torch.hub.load('s3prl/s3prl', 'wav2vec2')
```

### Ordered in publication date
Publication Date | Model | name | Paper | Input | Pre-train | Ckpt | Repo 
|---|---|---|---|---|---|---|---
10 Jul 2018 | CPC | cpc | [arxiv](https://arxiv.org/abs/1807.03748) | wave | [LibriLight-60k](https://github.com/facebookresearch/libri-light) | X | [FAIR](https://github.com/facebookresearch/CPC_audio)
5 Apr 2019 | APC | apc | [arxiv](https://arxiv.org/abs/1904.03240) | mel | [LibriSpeech-360](http://www.openslr.org/12) | O | [APC](https://github.com/iamyuanchung/Autoregressive-Predictive-Coding)
6 Apr 2019 | PASE | pase | [arxiv](https://arxiv.org/abs/1904.03416) | wave | [LibriSpeech-960](http://www.openslr.org/12) | X | [PASE](https://github.com/santi-pdp/pase)
11 Apr 2019 | wav2vec | wav2vec | [arxiv](https://arxiv.org/abs/1904.05862) | wave | [LibriSpeech-960](http://www.openslr.org/12) | O | [Fairseq](https://github.com/pytorch/fairseq)
12 Oct 2019 | vq-wav2vec | vq_wav2vec | [arxiv](https://arxiv.org/abs/1910.05453) | wave | [LibriSpeech-960](http://www.openslr.org/12) | O | [Fairseq](https://github.com/pytorch/fairseq)
25 Oct 2019 | Mockingjay | mockingjay | [arxiv](https://arxiv.org/abs/1910.12638) | mel | [LibriSpeech-960](http://www.openslr.org/12) | O | [S3PRL](https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning)
7 Feb 2020 | Modified-CPC | cpc | [arxiv](https://arxiv.org/abs/2002.02848) | wave | [LibriLight-60k](https://github.com/facebookresearch/libri-light) | O | [FAIR](https://github.com/facebookresearch/CPC_audio)
17 May 2020 | VQ-APC | vq_apc | [arxiv](https://arxiv.org/abs/2005.08392) | mel | [LibriSpeech-360](http://www.openslr.org/12) | O | [NPC](https://github.com/iamyuanchung/Autoregressive-Predictive-Coding)
18 May 2020 | Audio Albert | audio_albert | [arxiv](https://arxiv.org/abs/2005.08575) | mel | [LibriSpeech-960](http://www.openslr.org/12) | X | [S3PRL](https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning)
20 Jun 2020 | wav2vec 2.0 | wav2vec2 | [arxiv](https://arxiv.org/abs/2006.11477) | wave | [LibriSpeech-960](http://www.openslr.org/12) | O | [Fairseq](https://github.com/pytorch/fairseq)
12 Jul 2020 | TERA | tera | [arxiv](https://arxiv.org/abs/2007.06028) | mel | [LibriSpeech-960](http://www.openslr.org/12) | O | [S3PRL](https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning)
1 Nov 2020 | NPC | npc | [arxiv](https://arxiv.org/abs/2011.00406) | mel | [LibriSpeech-360](http://www.openslr.org/12) | O | [NPC](https://github.com/iamyuanchung/Autoregressive-Predictive-Coding)

### Ordered in different style of losses
- **Mockingjay**
    - Described in ["Mockingjay: Unsupervised Speech Representation Learning with Deep Bidirectional Transformer Encoders"](https://arxiv.org/abs/1910.12638)
    - *Transformer based, BERT-style masked reconstruction loss*
    - These papers used our implementations: [Adversarial Defense](https://arxiv.org/abs/2006.03214), [Understanding Self-attention](https://arxiv.org/abs/2006.03265)
    - Checkpoints are provided by this repo: [s3prl](https://github.com/s3prl/s3prl)
- **TERA**
    - Described in ["TERA: Self-Supervised Learning of Transformer Encoder Representation for Speech"](https://arxiv.org/abs/2007.06028)
    - *Transformer based, Advanced masked reconstruction loss*
    - Checkpoints are provided by this repo: [s3prl](https://github.com/s3prl/s3prl)
- **Audio ALBERT**
    - Described in ["Audio ALBERT: A Lite BERT for Self-supervised Learning of Audio Representation"](https://arxiv.org/abs/2005.08575)
    - *Transformer based, BERT-style masked reconstruction loss*
    - Checkpoints are provided by this repo: [s3prl](https://github.com/s3prl/s3prl)
- **CPC**
    - Described in ["Representation Learning with Contrastive Predictive Coding"](https://arxiv.org/abs/1807.03748)
    - *CNN based, InfoNCE contrastive loss*
    - Checkpoints are provided by this repo: [FAIR](https://github.com/facebookresearch/CPC_audio)
- **APC**
    - Described in ["An Unsupervised Autoregressive Model for Speech Representation Learning"](https://arxiv.org/abs/1904.03240)
    - *RNN based, unidirectional reconstruction loss*
    - Checkpoints are provided by this repo: [Alexander-H-Liu/NPC](https://github.com/Alexander-H-Liu/NPC)
- **VQ-APC**
    - Described in ["Vector-Quantized Autoregressive Predictive Coding"](https://arxiv.org/abs/2005.08392)
    - *RNN based, unidirectional reconstruction loss + vector quantization*
    - Checkpoints are provided by this repo: [Alexander-H-Liu/NPC](https://github.com/Alexander-H-Liu/NPC)
- **NPC**
    - Described in ["Non-Autoregressive Predictive Coding for Learning Speech Representations from Local Dependencies"](https://arxiv.org/abs/2011.00406)
    - *CNN based, reconstruction loss with Masked Convolution Blocks*
    - Checkpoints are provided by this repo: [Alexander-H-Liu/NPC](https://github.com/Alexander-H-Liu/NPC)
- **wav2vec**
    - Described in ["wav2vec: Unsupervised Pre-training for Speech Recognition"](https://arxiv.org/abs/1904.05862)
    - *CNN based, InfoNCE contrastive loss*
    - Checkpoints are provided by this repo: [fairseq](https://github.com/pytorch/fairseq/blob/master/examples/wav2vec/README.md)
- **vq-wav2vec**
    - Described in ["vq-wav2vec: Self-Supervised Learning of Discrete Speech Representations"](https://arxiv.org/abs/1910.05453)
    - *CNN based, InfoNCE contrastive loss*
    - Checkpoints are provided by this repo: [fairseq](https://github.com/pytorch/fairseq/blob/master/examples/wav2vec/README.md)
- **wav2vec 2.0**
    - Described in ["wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations"](https://arxiv.org/abs/2006.11477)
    - *CNN+Transformer based, InfoNCE contrastive loss + vector quantization + BERT-style masking*
    - Checkpoints are provided by this repo: [fairseq](https://github.com/pytorch/fairseq/blob/master/examples/wav2vec/README.md)
