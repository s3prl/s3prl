import logging

import yaml
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from omegaconf import OmegaConf
from fairseq.tasks.audio_pretraining import AudioPretrainingTask
from examples.data2vec.models.data2vec2 import (
    Data2VecMultiModel,
)  # register data2vec model into fairseq.task


SAMPLE_RATE = 16000
EXAMPLE_SEC = 5

logger = logging.getLogger(__name__)


class UpstreamExpert(torch.nn.Module):
    def __init__(self, ckpt, **kwds):
        super().__init__()

        ckpt = torch.load(ckpt, map_location="cpu")
        cfg = OmegaConf.create(ckpt["cfg"])

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        logger.info(yaml.dump(cfg_dict))

        task = AudioPretrainingTask.setup_task(cfg.task)
        self.model = task.build_model(cfg.model, from_checkpoint=True)
        self.model.load_state_dict(ckpt["model"])

        self.model.cfg.layerdrop = 0.0
        self.normalize = cfg.task.normalize

    def get_downsample_rates(self, key: str) -> int:
        return 320

    def forward(self, wavs):
        if self.normalize:
            wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        device = wavs[0].device
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)

        result = self.model(
            padded_wav,
            padding_mask=wav_padding_mask,
            mask=False,
            features_only=True,
        )
        hidden_states = result["layer_results"]

        return {
            "hidden_states": hidden_states,
        }
