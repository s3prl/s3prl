# Distiller (DistilHuBERT)

## Intro
This is the official implementation of [DistilHuBERT](https://arxiv.org/abs/2110.01900). Although the we only distilled represenations from HuBERT Base, the pretraining [code](/s3prl/pretrain) can learn from other SSL models.

## Get pre-trained models and inference representations
```
from s3prl.hub import distilhubert
...
```

## Benchmark DistilHuBERT on SUPERB
If you wish to benchmark [DistilHuBERT](https://arxiv.org/abs/2110.01900) with SUPERB, you can run with
```
python3 run_downstream.py -u distilhubert -s paper -d [task] -n ExpName
```

## Pretrain from scratch
Please refer to [pretrain/README.md](/s3prl/pretrain/README.md)

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
