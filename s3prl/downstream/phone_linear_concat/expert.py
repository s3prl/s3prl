# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ expert.py ]
#   Synopsis     [ the phone linear concat downstream wrapper ]
#   Author       [ Andy T. Liu (https://github.com/andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
from ..phone_linear.expert import DownstreamExpert as PhoneExpert
from .model import Model


class DownstreamExpert(PhoneExpert):
    """
    Basically the same as the phone linear expert
    """

    def __init__(self, upstream_dim, downstream_expert, **kwargs):
        super(DownstreamExpert, self).__init__(upstream_dim, downstream_expert, **kwargs)
        
        delattr(self, 'model')
        self.model = Model(input_dim=self.upstream_dim, output_class_num=self.train_dataset.class_num, **self.modelrc)
