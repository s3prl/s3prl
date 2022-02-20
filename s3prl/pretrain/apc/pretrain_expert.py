# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ pretrain/apc/pretrain_expert.py ]
#   Synopsis     [ the apc pretrain expert ]
#   Author       [ Andy T. Liu (https://github.com/andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import yaml
import copy
#-------------#
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
#-------------#
from pretrain.apc.dataset import ApcAudioDataset
from utility.audio import plot_spectrogram_to_numpy


####################
# UPSTREAM WRAPPER #
####################
class UpstreamPretrainExpert(nn.Module):
    """
    The APC pretrain expert
    """

    def __init__(self, datarc, upstream_config, device='cuda', multi_gpu=False, **kwargs):
        super(UpstreamPretrainExpert, self).__init__()

        self.datarc = datarc
        self.device = device
        self.multi_gpu = multi_gpu

        if type(upstream_config) == str:
            self.upstream_config = yaml.load(open(upstream_config, 'r'), Loader=yaml.FullLoader)
            print('[UpstreamPretrainExpert] - Using upstream config from:', upstream_config)
        elif type(upstream_config) == dict:
            self.upstream_config = upstream_config
            print('[UpstreamPretrainExpert] - Using upstream config from the previous experiment.')
        else:
            raise ValueError
        
        preprocessor = self._init_model()
        self._get_train_dataloader(preprocessor)

        if self.multi_gpu:
            self.model = torch.nn.DataParallel(self.model)
            print('[UpstreamPretrainExpert] - Multi-GPU training Enabled: ' + str(torch.cuda.device_count()))
        print('[UpstreamPretrainExpert] - Number of parameters: ' + str(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

    def _init_model(self):
        from upstream.apc.audio import create_transform
        from upstream.apc.apc import APC

        try:
            print('[UpstreamPretrainExpert] - Using the apc preprocessor, on-the-fly feature preprocessing')
            preprocessor, feat_dim = create_transform(copy.deepcopy(self.upstream_config['data']['audio']))
        except:
            raise NotImplementedError('Our upstream wrapper currently does not support other feature extracters, see: `s3prl/upstream/apc/expert.py`')
        
        print('[UpstreamPretrainExpert] - Initializing model...')
        self.model = APC(feat_dim, **self.upstream_config["model"]["paras"])
        self.n_future = self.upstream_config["task"]["n_future"]
        self.loss = torch.nn.L1Loss()
        return preprocessor

    def _get_train_dataloader(self, preprocessor):
        dataset = ApcAudioDataset(preprocessor,
                                  self.upstream_config['task'],
                                  self.datarc['train_batch_size'],
                                  **self.datarc)
        self.dataloader = DataLoader(dataset, batch_size=1, # for bucketing
                                     shuffle=True, num_workers=self.datarc['num_workers'],
                                     drop_last=False, pin_memory=True, collate_fn=dataset.collate_fn)

    # Interface
    def load_model(self, init_ckpt):
        assert 'model' in init_ckpt
        if self.multi_gpu:
            self.model.module.load_state_dict(init_ckpt['model'])
        else:
            self.model.load_state_dict(init_ckpt['model'])

    # Interface
    def loss_to_device(self):
        self.loss.to(self.device)

    # Interface
    def add_state_to_save(self, all_states):
        all_states['config'] = self.upstream_config
        all_states['model'] = self.model.state_dict() if not self.multi_gpu else \
                                 self.model.module.state_dict()
        all_states['Upstream_Config'] = self.upstream_config
        return all_states

    # Interface
    def get_train_dataloader(self):
        return self.dataloader

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
        
        # APC input = shifted target
        audio_len = [l-self.n_future for l in audio_len]
        pred_spec, _ = self.model(audio_feat[:,:-self.n_future,:], audio_len, testing=False)
        loss = self.loss(pred_spec, audio_feat[:,self.n_future:,:])

        if global_step % log_step == 0:
            spec_list = [pred_spec, audio_feat]
            name_list = ['pred_spec', 'true_spec']
            
            for i in range(len(spec_list)):
                spec = plot_spectrogram_to_numpy(spec_list[i][0].data.cpu().numpy())
                records[name_list[i]] = spec
            
        return loss, records

    # interface
    def on_before_zero_grad(self):
        pass
    
    # interface
    def log_records(self, records, logger, prefix, global_step, **kwargs):
        """
        Args:
            records:
                defaultdict(list), contents already appended

            logger:
                Tensorboard SummaryWriter
                please use f'{prefix}your_content_name' as key name
                to log your customized contents

            prefix:
                used to indicate downstream and train/test on Tensorboard
                eg. 'phone/train-'

            global_step:
                global_step in runner, which is helpful for Tensorboard logging
        """
        for key, values in records.items():
            logger.add_image(
                f'{prefix}{key}',
                values,
                global_step=global_step
            )