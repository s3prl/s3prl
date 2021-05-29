# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ expert.py ]
#   Synopsis     [ the speaker linear downstream wrapper ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import torch
from torch.nn.utils.rnn import pad_sequence
#-------------#
from ..speaker_linear_utter_libri.expert import DownstreamExpert as SpeakerExpert


class DownstreamExpert(SpeakerExpert):
    """
    Basically the same as the speaker utterance expert, except handles the speaker frame-wise label
    """

    def __init__(self, upstream_dim, downstream_expert, expdir, **kwargs):
        super(DownstreamExpert, self).__init__(upstream_dim, downstream_expert, expdir, **kwargs)

    # Interface
    def forward(self, mode, features, labels, records, **kwargs):
        """
        Args:
            features:
                list of unpadded features [feat1, feat2, ...]
                each feat is in torch.FloatTensor and already
                put in the device assigned by command-line args

            labels:
                the frame-wise spekaer labels

            records:
                defaultdict(list), by appending contents into records,
                these contents can be averaged and logged on Tensorboard
                later by self.log_records every log_step

        Return:
            loss:
                the loss to be optimized, should not be detached
        """
        lengths = torch.LongTensor([len(l) for l in features])

        features = pad_sequence(features, batch_first=True) # list of tensors -> tensors
        labels = labels.unsqueeze(-1).expand(features.size(0), features.size(1)).to(features.device)

        predicted = self.model(features)

        # cause logits are in (batch, seq, class) and labels are in (batch, seq)
        # nn.CrossEntropyLoss expect to have (N, class) and (N,) as input
        # here we flatten logits and labels in order to apply nn.CrossEntropyLoss
        class_num = predicted.size(-1)
        loss = self.objective(predicted.reshape(-1, class_num), labels.reshape(-1))

        predicted_classid = predicted.max(dim=-1).indices
        sames = (predicted_classid == labels)
        for s, l in zip(sames, lengths):
            records['acc'] += s[:l].tolist()

        return loss
