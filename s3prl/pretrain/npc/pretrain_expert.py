# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ pretrain/npc/pretrain_expert.py ]
#   Synopsis     [ the NPC pretrain expert ]
#   Author       [ Andy T. Liu (https://github.com/andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import copy
import torch
#-------------#
from utility.audio import plot_spectrogram_to_numpy
from pretrain.apc.pretrain_expert import UpstreamPretrainExpert as ApcPretrainExpert


####################
# UPSTREAM WRAPPER #
####################
class UpstreamPretrainExpert(ApcPretrainExpert):
    """
    The NPC pretrain expert
    """

    def __init__(self, datarc, upstream_config, device='cuda', multi_gpu=False, **kwargs):
        super(UpstreamPretrainExpert, self).__init__(datarc, upstream_config, device, multi_gpu, **kwargs)

    def _init_model(self):
        from upstream.npc.audio import create_transform
        from upstream.npc.npc import NPC

        try:
            print('[UpstreamPretrainExpert] - Using the apc preprocessor, on-the-fly feature preprocessing')
            preprocessor, feat_dim = create_transform(copy.deepcopy(self.upstream_config['data']['audio']))
        except:
            raise NotImplementedError('Our upstream wrapper currently does not support other feature extracters, see: `s3prl/upstream/apc/expert.py`')
        
        print('[UpstreamPretrainExpert] - Initializing model...')
        self.model = NPC(feat_dim, **self.upstream_config["model"]["paras"])
        self.loss = torch.nn.L1Loss(reduction='none')
        return preprocessor

    # Interface
    def forward(self, data, records={}, global_step=0, log_step=1000, **kwargs):
        """
        Args:
            data:
                [spec_masked, pos_enc, mask_label, attn_mask, spec_target]
            
            records:
                defaultdict(list), by appending contents into records,
                these contents can be averaged and logged on Tensorboard
                later by self.log_records every log_step

        Return:
            loss        
        """

        audio_feat, audio_len = data[0], data[1]
        audio_feat = audio_feat.to(self.device)
        
        # NPC: input = target
        pred_spec, _ = self.model(audio_feat)
        loss = self.loss(pred_spec, audio_feat)
        # Compute loss on valid part only
        effective_loss = 0
        for i,a_len in enumerate(audio_len):
            effective_loss += loss[i,:a_len,:].mean(dim=-1).sum()
        loss = effective_loss/sum(audio_len)

        if global_step % log_step == 0:
            spec_list = [pred_spec, audio_feat]
            name_list = ['pred_spec', 'true_spec']
            
            for i in range(len(spec_list)):
                spec = plot_spectrogram_to_numpy(spec_list[i][0].data.cpu().numpy())
                records[name_list[i]] = spec
            
        return loss, records