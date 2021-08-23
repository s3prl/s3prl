# MOS Prediction

Official Implementation of "Utilizing Self-supervised Representations for MOS Prediction", will be presented at INTERSPEECH 2021 [[arXiv](https://arxiv.org/abs/2104.03017)]

This code enables you to fine-tune a automatic Mean Opinion Score (MOS) Predictor with specific self-supervised upstream model.

## Example Usage
Use the following code to train a MOS Predictor with specific upstream model:
```python
EXP_NAME=mos_prediction_test_wav2vec2
UPSTREAM=wav2vec2
DOWNSTREAM=mos_prediction
python3 run_downstream.py -f -m train -n $EXP_NAME -u $UPSTREAM -d $DOWNSTREAM -o "config.downstream_expert.datarc.save_dir='result/downstream/${EXP_NAME}'"
```
If you only want to use the MOS predictor, please refer the code in the upstream folder (Check [**MOS Predictor**](../../upstream/mos_prediction)).

## Citation

If you find this MOS predictor useful, please consider citing following paper:
```
@article{tseng2021utilizing,
  title={Utilizing Self-supervised Representations for MOS Prediction},
  author={Tseng, Wei-Cheng and Huang, Chien-yu and Kao, Wei-Tsung and Lin, Yist Y and Lee, Hung-yi},
  journal={arXiv preprint arXiv:2104.03017},
  year={2021}
}
```
