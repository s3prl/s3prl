# Upstream Documentation

We organized most of the existing SSL pretrained models in [SUPERB Benchmark](https://arxiv.org/abs/2105.01051)'s framework.

## 1-for-all interface
We provide an all-in-one unified interface for numerous speech pretrained models.
**All the upstream models take input / output of the same format:**
- **input**: list of unpadded wavs `[wav1, wav2, ...]`, each wav is in `torch.FloatTensor`
- **output**: a dictionary where each key's corresponding value is either a padded sequence in `torch.FloatTensor` or a list of padded sequences, each in `torch.FloatTensor`. Every padded sequence is in the shape of `(batch_size, max_sequence_length_of_batch, hidden_size)`. At least a key `hidden_states` is available, which is a list.

For upstream models that operate on features other than wav (for example: log Mel, fbank, etc), the preprocessing of wav -> feature is done on-they-fly during model forward. Rest assured that this will not increase your runtime.

## How to use

The `Name` field in the [upstream information](#upstream-information) below is the string we use to specify different models. In other words, different upstream are identified with the exact string of their `Name`. The upstreams are loaded with pretrained weights.

### Use upstreams with our benchmark script

To evaluate upstreams with [SUPERB Benchmark](https://arxiv.org/abs/2105.01051), we provide a unified script for all upstreams: [run_downstream.py](../run_downstream.py). Please refer to [downstream/README.md](../downstream/README.md) for detailed usage.

#### Specify an upstream

In this script, we can use `-u` with the `Name` to switch different upstreams for benchmarking. Take **wav2vec 2.0 Base** for example:

```bash
python3 run_downstream.py -m train -u fbank -d example -n ExpName
python3 run_downstream.py -m train -u wav2vec2 -d example -n ExpName
```

#### Check all available upstreams

```bash
python3 run_downstream.py -h
```

### Use upstreams in your own project

After [installing s3prl](../../README.md#installation), you can use upstreams in your own codebase.

#### Specify an upstream

```python
import s3prl.hub as hub

model_0 = getattr(hub, 'fbank')()  # use classic FBANK
model_1 = getattr(hub, 'modified_cpc')()  # build the CPC model with pre-trained weights
model_2 = getattr(hub, 'tera')()  # build the TERA model with pre-trained weights
model_3 = getattr(hub, 'wav2vec2')()  # build the Wav2Vec 2.0 model with pre-trained weights

device = 'cuda'  # or cpu
model_3 = model_3.to(device)
wavs = [torch.randn(160000, dtype=torch.float).to(device) for _ in range(16)]
with torch.no_grad():
    reps = model_3(wavs)["hidden_states"]
```

#### Check all available upstreams

```python
import s3prl.hub as hub

print(dir(hub))
```

## Upstream Information

### SSL Upstreams

We support most of the existing SSL pretrained models. You can refer to [SUPERB](https://arxiv.org/abs/2105.01051) paper for their pre-training loss styles.

Publication Date | Model | Name | Paper | Input | Stride | Pre-train Data | Official Ckpt | Official Repo
|---|---|---|---|---|---|---|---|---
10 Jul 2018 | CPC | - | [arxiv](https://arxiv.org/abs/1807.03748) | wav | 10ms | - | X | Unavailable
5 Apr 2019 | APC | apc | [arxiv](https://arxiv.org/abs/1904.03240) | Mel | 10ms | [LibriSpeech-360](http://www.openslr.org/12) | O | [APC](https://github.com/Alexander-H-Liu/NPC)
6 Apr 2019 | PASE | pase_plus | [arxiv](https://arxiv.org/abs/1904.03416) | wav | 10ms | [LibriSpeech-960](http://www.openslr.org/12) | X | [PASE](https://github.com/santi-pdp/pase)
11 Apr 2019 | Wav2Vec | wav2vec | [arxiv](https://arxiv.org/abs/1904.05862) | wav | 10ms | [LibriSpeech-960](http://www.openslr.org/12) | O | [Fairseq](https://github.com/pytorch/fairseq)
12 Oct 2019 | VQ-Wav2Vec | vq_wav2vec | [arxiv](https://arxiv.org/abs/1910.05453) | wav | 10ms | [LibriSpeech-960](http://www.openslr.org/12) | O | [Fairseq](https://github.com/pytorch/fairseq)
25 Oct 2019 | Mockingjay | mockingjay | [arxiv](https://arxiv.org/abs/1910.12638) | Mel | 10ms | [LibriSpeech-960](http://www.openslr.org/12) | O | [S3PRL](https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning)
7 Feb 2020 | Modified-CPC | modified_cpc | [arxiv](https://arxiv.org/abs/2002.02848) | wav | 10ms | [LibriLight-60k](https://github.com/facebookresearch/libri-light) | O | [FAIR](https://github.com/facebookresearch/CPC_audio)
17 May 2020 | VQ-APC | vq_apc | [arxiv](https://arxiv.org/abs/2005.08392) | Mel | 10ms | [LibriSpeech-360](http://www.openslr.org/12) | O | [NPC](https://github.com/Alexander-H-Liu/NPC)
18 May 2020 | Audio Albert | audio_albert | [arxiv](https://arxiv.org/abs/2005.08575) | Mel | 10ms | [LibriSpeech-960](http://www.openslr.org/12) | X | [S3PRL](https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning)
20 Jun 2020 | Wav2Vec 2.0 | wav2vec2 / wav2vec2_large_ll60k | [arxiv](https://arxiv.org/abs/2006.11477) | wav | 20ms | [LibriSpeech-960](http://www.openslr.org/12) | O | [Fairseq](https://github.com/pytorch/fairseq)
12 Jul 2020 | TERA | tera | [arxiv](https://arxiv.org/abs/2007.06028) | Mel | 10ms | [LibriSpeech-960](http://www.openslr.org/12) | O | [S3PRL](https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning)
1 Nov 2020 | NPC | npc | [arxiv](https://arxiv.org/abs/2011.00406) | Mel | 10ms | [LibriSpeech-360](http://www.openslr.org/12) | X | [NPC](https://github.com/Alexander-H-Liu/NPC)
Jun 14 2021 | HuBERT | hubert / hubert_large_ll60k | [arxiv](https://arxiv.org/abs/2106.07447) | wav | 20ms | [LibriSpeech-960](http://www.openslr.org/12) | O | [Fairseq](https://github.com/pytorch/fairseq)
Dec 3 2019 | DeCoAR | decoar | [arxiv](https://arxiv.org/abs/1912.01679) | Mel | 10ms | [LibriSpeech-960](http://www.openslr.org/12) | O | [speech-representations](https://github.com/awslabs/speech-representations)
Dec 11 2020 | DeCoAR 2.0 | decoar2 | [arxiv](https://arxiv.org/abs/2012.06659) | Mel | 10ms | [LibriSpeech-960](http://www.openslr.org/12) | O | [speech-representations](https://github.com/awslabs/speech-representations)
Oct 5 2021 | DistilHuBERT | distilhubert | [arxiv](https://arxiv.org/abs/2110.01900) | wav | 20ms | [LibriSpeech-960](http://www.openslr.org/12) | O | [S3PRL](https://github.com/s3prl/s3prl)

### Acoustic Feature Upstreams

We also provide classic acoustic features as baselines. For each upstream with `Name`, you can configure their options (available by their `Backend`) in `s3prl/upstream/baseline/Name.yaml`.

| Feature | Name | Default Dim | Stride | Window | Backend |
| -------- | -------- | -------- | -------- | -------- | -------- |
| Spectrogram | spectrogram | 257 | 10ms | 25ms | [torchaudio-kaldi](https://pytorch.org/audio/stable/compliance.kaldi.html) |
| FBANK | fbank | 80 + delta1 + delta2 | 10ms | 25ms | [torchaudio-kaldi](https://pytorch.org/audio/stable/compliance.kaldi.html) |
| MFCC | mfcc | 13 + delta1 + delta2 | 10ms | 25ms | [torchaudio-kaldi](https://pytorch.org/audio/stable/compliance.kaldi.html) |
| Mel | mel | 80 | 10ms | 25ms | [torchaudio](https://pytorch.org/audio/stable/transforms.html) |
| Linear | linear | 201 | 10ms | 25ms | [torchaudio](https://pytorch.org/audio/stable/transforms.html) |

## Configure Upstreams

The upstreams can take two options `ckpt` and `model_config`, whose type are both `str`. You can refer to each upstream's [hubconf.py](./baseline/hubconf.py) and [expert.py](./baseline/expert.py) for their supported options. [Hubconf.py](./baseline/hubconf.py) under each upstream folder contains the entries you can use as the `Name` to initialize an upstream, which follows the protocol documented at [torch.hub.load](https://pytorch.org/docs/stable/hub.html). SSL upstreams with pretrained checkpoints typically has pre-registered `ckpt` at their [hubconf.py](./wav2vec2/hubconf.py) specifying the location of the pre-trained checkpoint. On the other hand, acoustic feature upstreams typically accept `model_config` as the configuration file for the feature extraction. Below is an example on how to pass options into an upstream entry to get different upstream instances.

```python
import torch
import s3prl.hub as hub

device = 'cuda' # or cpu
config_path = 's3prl/upstream/baseline/mfcc.yaml'
extracter = getattr(hub, 'baseline_local', model_config=config_path).to(device)

wavs = [torch.zeros(160000, dtype=torch.float).to(device) for _ in range(16)]
with torch.no_grad():
    mfcc = extracter(wavs)["hidden_states"]
```
