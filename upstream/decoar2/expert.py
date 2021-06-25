#-------------#
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
#-------------#
from .decoar2 import Decoar2
from .audio import create_transform
from collections import OrderedDict

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
        models = torch.load(ckpt)['model']
        self.model = Decoar2()
        component_state_dict = OrderedDict()
        for key in models.keys():
            component_state_dict[key] = models[key]
        self.model.load_state_dict(component_state_dict, strict=False)

        self.preprocessor = create_transform()

        self.output_dim = 768

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

        padding_mask = (
            torch.BoolTensor(features.shape).fill_(False).to(features.device)
        )

        for i in range(len(feat_lengths)):
            diff = feat_lengths[i] - size
            if diff == 0:
                continue
            padding_mask[i, diff:] = True

        features, layer_results = self.model(features, padding_mask)

        return {"default": features}
