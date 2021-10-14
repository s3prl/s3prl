# Distiller (DistilHuBERT)

## Intro
This is the official implementation of [DistilHuBERT](https://arxiv.org/abs/2110.01900). Although the we only distilled represenations from HuBERT Base, the pretraining [code](/s3prl/pretrain) can learn from other SSL models.

## Usage
The default extracted representations are `hidden_states`, which matches the default method proposed in our [paper](https://arxiv.org/abs/2110.01900). If you wish to use all the prediction heads' output representations, choose `all_hidden_states`.

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
