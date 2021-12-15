# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ pretrain/spec_augment/pretrain_expert.py ]
#   Synopsis     [ the spec augment transformer pretrain expert ]
#   Author       [ Andy T. Liu (https://github.com/andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
from torch.utils.data import DataLoader
#-------------#
from pretrain.mockingjay.pretrain_expert import UpstreamPretrainExpert as MockingjayPretrainExpert
from pretrain.spec_augment.dataset import KaldiAcousticDataset, OnlineAcousticDataset


####################
# UPSTREAM WRAPPER #
####################
class UpstreamPretrainExpert(MockingjayPretrainExpert):
    """
    The spec augment transformer pretrain expert
    """

    def __init__(self, datarc, upstream_config, device='cuda', multi_gpu=False, **kwargs):
        super(UpstreamPretrainExpert, self).__init__(datarc, upstream_config, device, multi_gpu, **kwargs)

    def _get_train_dataloader(self, extracter):
        if 'libri_root' in self.datarc and 'kaldi' not in self.upstream_config['audio']:
            dataset = OnlineAcousticDataset(extracter,
                                            self.upstream_config['task'],
                                            self.datarc['train_batch_size'],
                                            target_level=self.upstream_config['audio']['target_level'],
                                            **self.datarc)
        else:
            dataset = KaldiAcousticDataset(extracter,
                                           self.upstream_config['task'],
                                           self.datarc['train_batch_size'],
                                           **self.datarc)
        self.dataloader = DataLoader(dataset, batch_size=1, # for bucketing
                                     shuffle=True, num_workers=self.datarc['num_workers'],
                                     drop_last=False, pin_memory=True, collate_fn=dataset.collate_fn)
