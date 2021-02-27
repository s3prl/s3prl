# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ expert.py ]
#   Synopsis     [ the phone 1-hidden downstream wrapper ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
from ..timit_phone_linear.expert import DownstreamExpert as PhoneExpert
from .model import *


class DownstreamExpert(PhoneExpert):
    """
    Basically the same as the phone linear expert
    """

    def __init__(self, upstream_dim, downstream_expert, **kwargs):
        super(DownstreamExpert, self).__init__(upstream_dim, downstream_expert, **kwargs)
        
        delattr(self, 'model')
        model_cls = eval(self.modelrc['select'])
        model_conf = self.modelrc[self.modelrc['select']]
        self.model = model_cls(self.upstream_dim, output_class_num=self.train_dataset.class_num, **model_conf)
