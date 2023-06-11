import pickle
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from .models import fast_vgs, w2v2_model

logger = logging.getLogger(__name__)


class UpstreamExpert(torch.nn.Module):
    def __init__(self, ckpt, **kwargs):
        super().__init__()
        model_path = Path(ckpt)

        # load args
        with open(model_path / "args.pkl", "rb") as f:
            args = pickle.load(f)

        # load weights
        weights = torch.load(model_path / "best_bundle.pth")

        # if want to use the entire model for e.g. speech-image retrieval (need to first follow section 3 below)
        dual_encoder = fast_vgs.DualEncoder(args)
        cross_encoder = fast_vgs.CrossEncoder(args)
        dual_encoder.load_state_dict(weights['dual_encoder'])
        cross_encoder.load_state_dict(weights['cross_encoder'])

        # if only want to use the audio branch for e.g. feature extraction for speech downstream tasks
        # if you are loading fast-vgs features, it will say that weights of layer 8-11 (0-based) are not seed_dir, that's fine, because fast-vgs only has first 8 layers (i.e. layer 0-7) of w2v2 model, last four layers will be randomly initialized layers
        self.model = w2v2_model.Wav2Vec2Model_cls(args)
        self.model.carefully_load_state_dict(weights['dual_encoder']) # will filter out weights that don't belong to w2v2

    def get_downsample_rates(self, key: str) -> int:
        return 320

    def forward(self, wavs):
        device = wavs[0].device
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        wavs = [(wav - wav.mean()) / (wav.std() + 1e-8) for wav in wavs]
        padded_wav = pad_sequence(wavs, batch_first=True)

        # example of using the audio branch for feature extraction (model is a instance of w2v2_model.Wav2Vec2Model_cls), from layer 7 (0-based)
        model_out = self.model(source=padded_wav, padding_mask=wav_padding_mask, mask=False, features_only=True, superb=True)
        return model_out
