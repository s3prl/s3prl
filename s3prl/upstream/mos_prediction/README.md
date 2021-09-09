# MOS Prediction

Official Implementation of "Utilizing Self-supervised Representations for MOS Prediction", will be presented at INTERSPEECH 2021 [[arXiv](https://arxiv.org/abs/2104.03017)]

This code provides a automatic Mean Opinion Score (MOS) Predictor that utilize the self-supervised representations, implementing in PyTorch. The MOS Predictor is serves as an upstream model in s3prl toolkit.

## Example Usage
There are currently three available checkpoints that used different self-supervised speech models, namely: `mos_wav2vec2`, `mos_tera`, `mos_apc`.

And here's the example usage:
```python
import torch
from s3prl.hub import mos_wav2vec2

mos_predictor = mos_wav2vec2().cuda()
wavs = [torch.zeros(160000, dtype=torch.float).to(device) for _ in range(16)] # list of unpadded wavs `[wav1, wav2, ...]`, each wav is in `torch.FloatTensor`

with torch.no_grad():
    scores = mos_predictor(wavs)['scores'] # list of scores of the wavs `[score1, score2, ...]`
```

## Train Your Own MOS Predictor
You can also train your own MOS Predictor with different self-supervised speech model by using the code in the downstream folder (Check [**Train your own MOS Predictor**](../../downstream/mos_prediction)).

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
