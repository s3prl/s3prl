# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ pretrain/byol/pretrain_expert.py ]
#   Synopsis     [ the byol pretrain expert ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import yaml
#-------------#
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
#-------------#
from pretrain.byol.byol_learner import BYOL
from upstream.byol.model import AudioEncoder
from upstream.baseline.preprocessor import get_preprocessor
#-------------#
from pretrain.byol.dataset import OnlineAcousticDataset
from utility.audio import plot_spectrogram_to_numpy


####################
# UPSTREAM WRAPPER #
####################
class UpstreamPretrainExpert(nn.Module):
    """
    The BYOL pretrain expert
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
        
        assert 'libri_root' in self.datarc
        print('[UpstreamPretrainExpert] - Using online preprocessor, on-the-fly feature extraction')
        extracter, input_dim, _ = get_preprocessor(self.upstream_config['audio'])
        print('[UpstreamPretrainExpert] - Input dim:', input_dim)

        self._get_train_dataloader(extracter)

        print('[UpstreamPretrainExpert] - Initializing model...')
        #self.model = AudioNTT2020(n_mels=input_dim, d=2048)
        self.model = AudioEncoder(input_dim, pretrain=True, **self.upstream_config['audio_encoder'])

        if self.multi_gpu:
            self.model = torch.nn.DataParallel(self.model)
            print('[UpstreamPretrainExpert] - Multi-GPU training Enabled: ' + str(torch.cuda.device_count()))
        print('[UpstreamPretrainExpert] - Number of parameters: ' + str(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

        self.learner = BYOL(self.model, 
                            image_size=[self.upstream_config['task']['sequence_length'], input_dim], # Shape of log-mel spectrogram [T, F]
                            hidden_layer=-1,
                            projection_size=256,
                            projection_hidden_size=4096,
                            moving_average_decay=0.99,
                            channels=0,
                            use_momentum=self.upstream_config['task']['use_momentum'])

    def _get_train_dataloader(self, extracter):
        assert 'libri_root' in self.datarc and 'audio' in self.upstream_config
        dataset = OnlineAcousticDataset(extracter,
                                        self.upstream_config['task'],
                                        self.datarc['train_batch_size'],
                                        target_level=self.upstream_config['audio']['target_level'],
                                        **self.datarc)
        self.dataloader = DataLoader(dataset, batch_size=1, # for bucketing
                                     shuffle=True, num_workers=self.datarc['num_workers'],
                                     drop_last=False, pin_memory=True, collate_fn=dataset.collate_fn)

    # Interface
    def load_model(self, all_states):
        if self.multi_gpu:
            self.model.module.load_state_dict(all_states['Model'])
        else:
            self.model.load_state_dict(all_states['Model'])

    # Interface
    def add_state_to_save(self, all_states):
        all_states['Model'] = self.model.state_dict() if not self.multi_gpu else \
                                 self.model.module.state_dict()
        all_states['Config'] = self.upstream_config
        return all_states

    # Interface
    def get_train_dataloader(self):
        return self.dataloader

    # Interface
    def forward(self, data, records={}, global_step=0, log_step=1000, **kwargs):
        """
        Args:
            data:
                [spec_stacked]
            
            records:
                defaultdict(list), by appending contents into records,
                these contents can be averaged and logged on Tensorboard
                later by self.log_records every log_step

        Return:
            loss        
        """
        
        origin = data[0] # (B, T, F)
        view1 = data[1].to(self.device) # (B, T, F)
        view2 = data[2].to(self.device) # (B, T, F)
        loss = self.learner(view1, view2)

        if global_step % log_step == 0:
            spec_list = [origin, \
                         view1[:, :, :], \
                         view2[:, :, :]]
            name_list = ['spec', 'view1', 'view2']
            
            for i in range(len(spec_list)):
                spec = plot_spectrogram_to_numpy(spec_list[i][0].data.cpu().numpy())
                records[name_list[i]] = spec
            
        return loss, records

    # interface
    def on_before_zero_grad(self):
        if self.upstream_config['task']['use_momentum']:
            self.learner.update_moving_average()

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