# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/npc/vq.py ]
#   Synopsis     [ the vq component for npc]
#   Author       [ S3PRL ]
#   Reference    [ VQ-layer modified from
#                  https://github.com/iamyuanchung/VQ-APC/blob/283d338/vqapc_model.py ]
"""*********************************************************************************************"""


import numpy as np

###############
# IMPORTATION #
###############
import torch
import torch.nn as nn
from torch.nn.functional import gumbel_softmax

############
# CONSTANT #
############
EPS = 1e-10


############
# VQ LAYER #
############
class VQLayer(nn.Module):
    def __init__(self, input_size, codebook_size, code_dim, gumbel_temperature):
        """
        Defines a VQ layer that follows an RNN layer.
        input_size: an int indicating the pre-quantized input feature size,
            usually the hidden size of RNN.
        codebook_size: an int indicating the number of codes.
        code_dim: an int indicating the size of each code. If not the last layer,
            then must equal to the RNN hidden size.
        gumbel_temperature: a float indicating the temperature for gumbel-softmax.
        """
        super(VQLayer, self).__init__()
        # Directly map to logits without any transformation.
        self.codebook_size = codebook_size
        self.vq_logits = nn.Linear(input_size, codebook_size)
        self.gumbel_temperature = gumbel_temperature
        self.codebook_CxE = nn.Linear(codebook_size, code_dim, bias=False)
        self.token_usg = np.zeros(codebook_size)

    def forward(self, inputs_BxLxI, testing, lens=None):
        logits_BxLxC = self.vq_logits(inputs_BxLxI)
        if testing:
            # During inference, just take the max index.
            shape = logits_BxLxC.size()
            _, ind = logits_BxLxC.max(dim=-1)
            onehot_BxLxC = torch.zeros_like(logits_BxLxC).view(-1, shape[-1])
            onehot_BxLxC.scatter_(1, ind.view(-1, 1), 1)
            onehot_BxLxC = onehot_BxLxC.view(*shape)
        else:
            onehot_BxLxC = gumbel_softmax(
                logits_BxLxC, tau=self.gumbel_temperature, hard=True, eps=EPS, dim=-1
            )
            self.token_usg += (
                onehot_BxLxC.detach()
                .cpu()
                .reshape(-1, self.codebook_size)
                .sum(dim=0)
                .numpy()
            )
        codes_BxLxE = self.codebook_CxE(onehot_BxLxC)

        return logits_BxLxC, codes_BxLxE

    def report_ppx(self):
        """Computes perplexity of distribution over codebook"""
        acc_usg = self.token_usg / sum(self.token_usg)
        return 2 ** sum(-acc_usg * np.log2(acc_usg + EPS))

    def report_usg(self):
        """Computes usage each entry in codebook"""
        acc_usg = self.token_usg / sum(self.token_usg)
        # Reset
        self.token_usg = np.zeros(self.codebook_size)
        return acc_usg
