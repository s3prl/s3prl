# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/apc/apc.py ]
#   Synopsis     [ the apc and vq-apc model ]
#   Author       [ iamyuanchung ]
#   Reference    [ https://github.com/iamyuanchung/VQ-APC/blob/283d338/vqapc_model.py ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import copy
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
#-------------#
from .vq import VQLayer


#######
# APC #
#######
class APC(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, residual, vq=None):
        """
            input_size: an int indicating the input feature size, e.g., 80 for Mel.
            hidden_size: an int indicating the RNN hidden size.
            num_layers: an int indicating the number of RNN layers.
            dropout: a float indicating the RNN dropout rate.
            residual: a bool indicating whether to apply residual connections.
        """
        super(APC, self).__init__()

        assert num_layers > 0
        self.hidden_size = hidden_size
        self.code_dim = hidden_size # ToDo: different size?
        self.num_layers = num_layers
        in_sizes = [input_size] + [hidden_size] * (num_layers - 1)
        out_sizes = [hidden_size] * num_layers
        self.rnn_layers = nn.ModuleList(
            [nn.GRU(input_size=in_size, hidden_size=out_size, batch_first=True)
            for (in_size, out_size) in zip(in_sizes, out_sizes)])

        self.rnn_dropout = nn.Dropout(dropout)

        self.rnn_residual = residual

        #  Create N-group VQ layers (Last layer only)
        self.apply_vq = vq is not None
        if self.apply_vq:
            self.vq_layers = []
            vq_config = copy.deepcopy(vq)
            codebook_size = vq_config.pop('codebook_size')
            self.vq_code_dims = vq_config.pop('code_dim')
            assert len(self.vq_code_dims)==len(codebook_size)
            assert sum(self.vq_code_dims)==hidden_size
            for cs,cd in zip(codebook_size,self.vq_code_dims):
                self.vq_layers.append(VQLayer(input_size=cd,
                                              code_dim=cd,
                                              codebook_size=cs,
                                              **vq_config))
            self.vq_layers = nn.ModuleList(self.vq_layers)

        # TODO: Start with a high temperature and anneal to a small one.
        # Final regression layer
        self.postnet = nn.Linear(hidden_size, input_size)

    def create_msg(self):
        msg_list = []
        msg_list.append('Model spec.| Method = APC\t| Apply VQ = {}\t'\
                        .format(self.apply_vq))
        msg_list.append('           | n layers = {}\t| Hidden dim = {}'\
                        .format(self.num_layers, self.hidden_size))
        return msg_list
    
    def report_ppx(self):
        if self.apply_vq:
            # ToDo: support more than 2 groups
            ppx = [m.report_ppx() for m in self.vq_layers] + [None]
            return ppx[0], ppx[1]
        else:
            return None, None
    
    def report_usg(self):
        if self.apply_vq:
            # ToDo: support more than 2 groups
            usg = [m.report_usg() for m in self.vq_layers] + [None]
            return usg[0], usg[1]
        else:
            return None, None

    def forward(self, frames_BxLxM, seq_lengths_B, testing):
        """
        Input:
            frames_BxLxM: a 3d-tensor representing the input features.
            seq_lengths_B: sequence length of frames_BxLxM.
            testing: a bool indicating training or testing phase.
        Return:
            predicted_BxLxM: the predicted output; used for training.
            hiddens_NxBxLxH: the RNN hidden representations across all layers.
        """
        max_seq_len = frames_BxLxM.size(1)

        # N is the number of RNN layers.
        hiddens_NxBxLxH = []

        # RNN
        # Prepare initial packed RNN input.
        packed_rnn_inputs = pack_padded_sequence(frames_BxLxM, seq_lengths_B,
                                                 batch_first=True,
                                                 enforce_sorted=False)
        for i, rnn_layer in enumerate(self.rnn_layers):
            # https://discuss.pytorch.org/t/rnn-module-weights-are-not-part-of-single-contiguous-chunk-of-memory/6011/14
            rnn_layer.flatten_parameters()
            packed_rnn_outputs, _ = rnn_layer(packed_rnn_inputs)

            # Unpack RNN output of current layer.
            rnn_outputs_BxLxH, _ = pad_packed_sequence(packed_rnn_outputs,
                                                       batch_first=True,
                                                       total_length=max_seq_len)
            # Apply dropout to output.
            rnn_outputs_BxLxH = self.rnn_dropout(rnn_outputs_BxLxH)

            # Apply residual connections.
            if self.rnn_residual and i > 0:
                # Unpack the original input.
                rnn_inputs_BxLxH, _ = pad_packed_sequence(packed_rnn_inputs,
                                                          batch_first=True,
                                                          total_length=max_seq_len)
                rnn_outputs_BxLxH += rnn_inputs_BxLxH

            hiddens_NxBxLxH.append(rnn_outputs_BxLxH)

            # VQ at last layer only
            if self.apply_vq and (i == len(self.rnn_layers)-1):
                q_feat = []
                offet = 0
                for vq_layer,cd in zip(self.vq_layers,self.vq_code_dims):
                    _, q_f = vq_layer(rnn_outputs_BxLxH[:,:,offet:offet+cd], testing)
                    q_feat.append(q_f)
                    offet += cd
                rnn_outputs_BxLxH = torch.cat(q_feat,dim=-1)

            # Prepare packed input for the next layer.
            # Note : enforce sorted = False might lead to CUDNN_STATUS_EXECUTION_FAILED
            if i < len(self.rnn_layers)-1:
                packed_rnn_inputs = pack_padded_sequence(rnn_outputs_BxLxH,
                                                         seq_lengths_B, batch_first=True,
                                                         enforce_sorted=False)
        # Only return last layer feature
        feature = hiddens_NxBxLxH[-1]

        # Generate final output from codes.
        predicted_BxLxM = self.postnet(rnn_outputs_BxLxH)
        return predicted_BxLxM, feature