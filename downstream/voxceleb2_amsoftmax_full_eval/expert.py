# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ expert.py ]
#   Synopsis     [ the phone linear downstream wrapper ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import math
import torch
import random
import kaldi_io
import numpy as np
#-------------#
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
#-------------#
from .model import Model, AMSoftmaxLoss, SoftmaxLoss, UtteranceExtractor
from .dataset import SpeakerVerifi_train, SpeakerVerifi_dev, SpeakerVerifi_test, SpeakerVerifi_plda
from argparse import Namespace
from .utils import EER, compute_metrics
import IPython
import pdb

def decide_utter_input_dim(agg_module_name, modelrc):
    if agg_module_name =="ASP":
        utter_input_dim = modelrc['input_dim']*2
    elif agg_module_name == "SP":
        # after aggregate to utterance vector, the vector hidden dimension will become 2 * aggregate dimension.
        utter_input_dim = modelrc['agg_dim']*2
    elif agg_module_name == "MP":
        utter_input_dim = modelrc['agg_dim']
    else:
        utter_input_dim = modelrc['input_dim']
    return utter_input_dim

class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log

    Note 1.
        dataloaders should output in the following format:

        [[wav1, wav2, ...], your_other_contents, ...]

        where wav1, wav2 ... are in variable length
        and wav1 is in torch.FloatTensor
    """

    def __init__(self, upstream_dim, downstream_expert, evaluate_split, expdir, **kwargs):
        super(DownstreamExpert, self).__init__()
        # config
        self.upstream_dim = upstream_dim
        self.downstream = downstream_expert
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']

        #############################################################################################

        # dataset
        train_config = {"vad_config":self.datarc['vad_config'], "file_path": [self.datarc['dev_root']], 
                        "key_list":["Voxceleb1"], "meta_data": self.datarc['train_meta_data'], 
                        "max_timestep": self.datarc["max_timestep"]}

        self.train_dataset = SpeakerVerifi_train(**train_config)

        dev_config = {"vad_config":self.datarc['vad_config'], "file_path": [self.datarc['dev_root']], 
            "meta_data": self.datarc['dev_meta_data']}
        
        self.dev_dataset = SpeakerVerifi_dev(**dev_config)

        test_config = {"vad_config":self.datarc['vad_config'], "file_path": [self.datarc['test_root']], 
            "meta_data": self.datarc['test_meta_data']}
        
        self.test_dataset = SpeakerVerifi_test(**test_config)

        train_plda_config = {"vad_config":self.datarc['vad_config'], "file_path": [self.datarc['dev_root']], 
            "key_list":["Voxceleb1_train_plda"], "meta_data": self.datarc['dev_meta_data']}

        self.train_dataset_plda = SpeakerVerifi_plda(**train_plda_config)

        test_plda_config = {"vad_config":self.datarc['vad_config'], "file_path": [self.datarc['test_root']], 
            "key_list":["Voxceleb1_test_plda"], "meta_data": self.datarc['dev_meta_data']}

        self.test_dataset_plda = SpeakerVerifi_plda(**test_plda_config)


        #########################################################################################################
        
        # module
        self.connector = nn.Linear(self.upstream_dim, self.modelrc['input_dim'])
        
        # downstream model
        if self.modelrc["module_config"][self.modelrc['module']].get("agg_dim"):
            self.modelrc['agg_dim'] = self.modelrc["module_config"][self.modelrc['module']]["agg_dim"]
        else:
            self.modelrc['agg_dim'] = self.modelrc['input_dim']
        
        ModelConfig = {"input_dim": self.modelrc['input_dim'], "agg_dim": self.modelrc['agg_dim'],
                       "agg_module": self.modelrc['agg_module'], "module": self.modelrc['module'], 
                       "hparams": self.modelrc["module_config"][self.modelrc['module']]}
        
        # downstream model extractor include aggregation module
        self.model = Model(**ModelConfig)

        utter_input_dim = decide_utter_input_dim(self.modelrc["agg_module"], self.modelrc)

        # after extract utterance level vector, put it to utterance extractor (XVector Architecture)
        self.utterance_extractor= UtteranceExtractor(utter_input_dim, self.modelrc['input_dim'])

        # SoftmaxLoss loss or AMSoftmaxLoss
        self.objective = eval(self.modelrc['ObjectiveLoss'])(speaker_num=self.train_dataset.speaker_num, 
                                                            **self.modelrc['LossConfig'][self.modelrc['ObjectiveLoss']])
        # utils
        self.score_fn  = nn.CosineSimilarity(dim=-1)
        self.eval_metric = EER

        if evaluate_split in ['train_plda', 'test_plda']:
            self.ark = open(f'{expdir}/{evaluate_split}.rep.ark', 'wb')

    # Interface
    def get_dataloader(self, mode):
        """
        Args:
            mode: string
                'train', 'dev' or 'test'

        Return:
            a torch.utils.data.DataLoader returning each batch in the format of:

            [wav1, wav2, ...], your_other_contents1, your_other_contents2, ...

            where wav1, wav2 ... are in variable length
            each wav is torch.FloatTensor in cpu with:
                1. dim() == 1
                2. sample_rate == 16000
                3. directly loaded by torchaudio
        """

        if mode == 'train':
            return self._get_train_dataloader(self.train_dataset)            
        elif mode == 'dev':
            return self._get_eval_dataloader(self.dev_dataset)
        elif mode == 'test':
            return self._get_eval_dataloader(self.test_dataset)
        elif mode == "train_plda":
            return self._get_eval_dataloader(self.train_dataset_plda) 
        elif mode == "test_plda":
            return self._get_eval_dataloader(self.test_dataset_plda)

    def _get_train_dataloader(self, dataset):
        return DataLoader(
            dataset, batch_size=self.datarc['train_batch_size'], 
            shuffle=True, num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset, batch_size=self.datarc['eval_batch_size'],
            shuffle=False, num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn
        )    


    # Interface
    def get_train_dataloader(self):
        return self._get_train_dataloader(self.train_dataset)

    # Interface
    def get_dev_dataloader(self):
        return self._get_eval_dataloader(self.dev_dataset)

    # Interface
    def get_test_dataloader(self):
        return self._get_eval_dataloader(self.test_dataset)

    # Interface
    def forward(self, mode, features, utter_idx, labels, records, **kwargs):
        """
        Args:
            features:
                the features extracted by upstream
                put in the device assigned by command-line args

            labels:
                the speaker labels

            records:
                defaultdict(list), by appending scalars into records,
                these scalars will be averaged and logged on Tensorboard

            logger:
                Tensorboard SummaryWriter, given here for logging/debugging
                convenience, please use "self.downstream/your_content_name" as key
                name to log your customized contents

            global_step:
                global_step in runner, which is helpful for Tensorboard logging

        Return:
            loss:
                the loss to be optimized, should not be detached
        """

        features_pad = pad_sequence(features, batch_first=True)
        
<<<<<<< HEAD
        if self.modelrc['module'] == "XVector":
            # since XVector will substract total sequence length, we directly substract 14.
=======
        if self.modelrc['module'] == 'XVector':
>>>>>>> origin/voxceleb1_tdnn
            attention_mask = [torch.ones((feature.shape[0]-14)) for feature in features]
        else:
            attention_mask = [torch.ones((feature.shape[0])) for feature in features]

        attention_mask_pad = pad_sequence(attention_mask,batch_first=True)
        attention_mask_pad = (1.0 - attention_mask_pad) * -100000.0

        features_pad = self.connector(features_pad)
        agg_vec = self.model(features_pad, attention_mask_pad.cuda())

        if mode == 'train':
            labels = torch.LongTensor(labels).to(features_pad.device)
            agg_vec = self.utterance_extractor(agg_vec)
            loss = self.objective(agg_vec, labels)
            
            return loss
        
        elif mode in ['dev', 'test']:
            # normalize to unit vector 
            agg_vec = agg_vec / (torch.norm(agg_vec, dim=-1).unsqueeze(-1))

            # separate batched data to pair data.
            vec1, vec2 = self.separate_data(agg_vec, labels)
            scores = self.score_fn(vec1,vec2).squeeze().cpu().detach().tolist()
            ylabels = torch.stack(labels).cpu().detach().long().tolist()

            if len(ylabels) > 1:
                records['scores'].extend(scores)
                records['ylabels'].extend(ylabels)
            else:
                records['scores'].append(scores)
                records['ylabels'].append(ylabels)
            return torch.tensor(0)
        
        elif mode in ['train_plda', 'test_plda']:
            for key, vec in zip(utter_idx, agg_vec):
                vec = vec.view(-1).detach().cpu().numpy()
                kaldi_io.write_vec_flt(self.ark, vec, key=key)

    # interface
    def log_records(self, mode, records, logger, global_step, batch_ids, total_batch_num, **kwargs):
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
        if mode in ['dev', 'test']:

            EER_result =self.eval_metric(np.array(records['ylabels']), np.array(records['scores']))

            records['EER'] = EER_result[0]

            logger.add_scalar(
                f'{mode}-'+'EER',
                records['EER'],
                global_step=global_step
            )
            print(f'{mode} ERR: {records["EER"]}')
        
        elif mode in ['train_plda', 'test_plda']:
            self.ark.close()
        
    def separate_data(self, agg_vec, ylabel):

        total_num = len(ylabel) 
        feature1 = agg_vec[:total_num]
        feature2 = agg_vec[total_num:]
        
        return feature1, feature2
