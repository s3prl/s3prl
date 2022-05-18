import logging
import re
from collections import OrderedDict

# -------------#
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from .audio import create_transform

# -------------#
from .decoar import Decoar

############
# CONSTANT #
############
EXAMPLE_FEAT_SEQLEN = 1000


###################
# UPSTREAM EXPERT #
###################
class UpstreamExpert(nn.Module):
    """
    The APC wrapper
    """

    def __init__(self, ckpt, **kwargs):
        super(UpstreamExpert, self).__init__()
        models = torch.load(ckpt)["model"]
        self.model = Decoar()
        component_state_dict = OrderedDict()

        def name_convert(key: str):
            result = re.search("(forward|backward)_lstm\.(.+)(\d+)", key)
            if result is not None:
                direction, name, layer = result.groups()
                new_name = f"{direction}_lstms.{layer}.{name}0"
                logging.debug(f"{key} -> {new_name}")
                return new_name
            else:
                return key

        for key in models.keys():
            component_state_dict[name_convert(key)] = models[key]
        self.model.load_state_dict(component_state_dict, strict=False)

        self.preprocessor = create_transform()

        self.output_dim = 2048

    def get_downsample_rates(self, key: str) -> int:
        return 160

    def forward(self, wavs):
        """
        Args:
            wavs:
                list of unpadded wavs [wav1, wav2, ...]
                each wav is in torch.FloatTensor with sample rate 16000
                and already put in the device assigned by command-line args

        Return:
            features:
                list of unpadded features [feat1, feat2, ...]
                each feat is in torch.FloatTensor and already
                put in the device assigned by command-line args
        """
        features = [self.preprocessor(wav.unsqueeze(0)) for wav in wavs]
        feat_lengths = [len(feat) for feat in features]
        size = max(feat_lengths)
        features = pad_sequence(features, batch_first=True)

        padding_mask = torch.BoolTensor(features.shape).fill_(False).to(features.device)

        for i in range(len(feat_lengths)):
            diff = feat_lengths[i] - size
            if diff == 0:
                continue
            padding_mask[i, diff:] = True

        layer_features = self.model(features, padding_mask)
        return {
            "hidden_states": layer_features,
            "last_hidden_state": layer_features[-1],
        }

        # This forward function only does the model forward
        # The return dict is then handled by UpstreamBase's hooks
