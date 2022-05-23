import argparse
from typing import List
from packaging import version

import torch
import fairseq
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from ..interfaces import UpstreamBase
from s3prl.utility.helper import zero_mean_unit_var_norm
from . import model


class UpstreamExpert(UpstreamBase):
    def __init__(self, ckpt, **kwargs):
        super().__init__(**kwargs)
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [ckpt], strict=False
        )
        self.model = model[0]
        self.wav_normalize = cfg.task.normalize

        if len(self.hooks) == 0:
            module_name = "self.model.encoder.layers"
            for module_id in range(len(eval(module_name))):
                self.add_hook(
                    f"{module_name}[{module_id}]",
                    lambda input, output: input[0].transpose(0, 1),
                )
            self.add_hook("self.model.encoder", lambda input, output: output[0])

            def postprocess(xs):
                names, hiddens = zip(*xs)
                unpad_len = min([hidden.size(1) for hidden in hiddens])
                hiddens = [hidden[:, :unpad_len, :] for hidden in hiddens]
                return list(zip(names, hiddens))

            self.hook_postprocess = postprocess

    def get_downsample_rates(self, key: str) -> int:
        return 320

    def forward(self, wavs):
        device = wavs[0].device
        if self.wav_normalize:
            wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)

        results = self.model.extract_features(padded_wav, wav_padding_mask)

        # This forward function only does the model forward
        # The return dict is then handled by UpstreamBase's hooks
