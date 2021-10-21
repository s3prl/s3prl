# Distiller (DistilHuBERT)

## Intro
This is the official implementation of [DistilHuBERT](https://arxiv.org/abs/2110.01900). Although the we only distilled represenations from HuBERT Base, the pretraining [code](../../pretrain/README.md) can learn from other SSL models.

## Get pre-trained models and inference representations

```python
import torch
from s3prl.hub import distilhubert

wavs = [torch.randn(16000) for _ in range(4)]
pretrained_model = distilhubert()
results = pretrained_model(wavs)

# The representation used in the paper
representation = results["paper"]

# All hidden states
hidden_states = results["hidden_states"]
```

## Pretrain from scratch
Please refer to [pretrain/README.md](../../pretrain/README.md)

## Benchmark DistilHuBERT on SUPERB

If you wish to evaluate [DistilHuBERT](https://arxiv.org/abs/2110.01900) on [SUPERB Benchmark](https://arxiv.org/abs/2105.01051), you can follow the [run_downstream.py](../../downstream/README.md) documentation:

```bash
python3 run_downstream.py -u distilhubert -s paper -d [task] -n ExpName
```

## Citation
If you use any of the pre-trained models related DistilHuBERT in this repository, please cite our paper:
```
@article{chang2021distilhubert,
  title={{DistilHuBERT}: Speech Representation Learning by Layer-wise Distillation of Hidden-unit {BERT}},
  author={Chang, Heng-Jui and Yang, Shu-wen and Lee, Hung-yi},
  journal={arXiv preprint arXiv:2110.01900},
  year={2021}
}
```
