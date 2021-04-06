# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ expert.py ]
#   Synopsis     [ the speech separation downstream wrapper ]
#   Source       [ Reference some code from https://github.com/funcwj/uPIT-for-speech-separation and https://github.com/asteroid-team/asteroid ]
#   Author       [ Zili Huang ]
#   Copyright    [ Copyright(c), Johns Hopkins University ]
"""*********************************************************************************************"""

###############
# IMPORTATION #
###############
import os
import math
import random
import h5py
import numpy as np
from collections import defaultdict
import librosa

# -------------#
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_sequence, pad_sequence
import torch.nn.functional as F

# -------------#
from .model import SepRNN
from .dataset import SeparationDataset
from asteroid.metrics import get_metrics
from .loss import MSELoss, SISDRLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#COMPUTE_METRICS = ["si_sdr", "sdr", "sir", "sar", "stoi"]
COMPUTE_METRICS = ["si_sdr"]

def match_length(feat_list, length_list):
    assert len(feat_list) == len(length_list)
    bs = len(length_list)
    new_feat_list = []
    for i in range(bs):
        assert abs(feat_list[i].size(0) - length_list[i]) < 5
        if feat_list[i].size(0) == length_list[i]:
            new_feat_list.append(feat_list[i])
        elif feat_list[i].size(0) > length_list[i]:
            new_feat_list.append(feat_list[i][:length_list[i], :])
        else:
            new_feat = torch.zeros(length_list[i], feat_list[i].size(1)).to(feat_list[i].device)
            new_feat[:feat_list[i].size(0), :] = feat_list[i]
            new_feat_list.append(new_feat)
    return new_feat_list

class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, downstream_expert, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert["datarc"]
        self.loaderrc = downstream_expert["loaderrc"]
        self.modelrc = downstream_expert["modelrc"]

        self.train_dataset = SeparationDataset(
                data_dir=self.loaderrc["train_dir"],
                rate=self.datarc['rate'],
                src=self.datarc['src'],
                tgt=self.datarc['tgt'],
                n_fft=self.datarc['n_fft'],
                hop_length=self.datarc['hop_length'],
                win_length=self.datarc['win_length'],
                window=self.datarc['window'],
                center=self.datarc['center'],
            )
        self.dev_dataset = SeparationDataset(
                data_dir=self.loaderrc["dev_dir"],
                rate=self.datarc['rate'],
                src=self.datarc['src'],
                tgt=self.datarc['tgt'],
                n_fft=self.datarc['n_fft'],
                hop_length=self.datarc['hop_length'],
                win_length=self.datarc['win_length'],
                window=self.datarc['window'],
                center=self.datarc['center'],
        )
        self.test_dataset = SeparationDataset(
                data_dir=self.loaderrc["test_dir"],
                rate=self.datarc['rate'],
                src=self.datarc['src'],
                tgt=self.datarc['tgt'],
                n_fft=self.datarc['n_fft'],
                hop_length=self.datarc['hop_length'],
                win_length=self.datarc['win_length'],
                window=self.datarc['window'],
                center=self.datarc['center'],
        )

        if self.modelrc["model"] == "SepRNN":
            self.model = SepRNN(
                input_dim=self.upstream_dim,
                num_bins=int(self.datarc['n_fft'] / 2 + 1),
                rnn=self.modelrc["rnn"],
                num_spks=self.datarc['num_speakers'],
                num_layers=self.modelrc["rnn_layers"],
                hidden_size=self.modelrc["hidden_size"],
                dropout=self.modelrc["dropout"],
                non_linear=self.modelrc["non_linear"],
                bidirectional=self.modelrc["bidirectional"]
            )
        else:
            raise ValueError("Model type not defined.")

        self.loss_type = self.modelrc["loss_type"]
        if self.modelrc["loss_type"] == "MSE":
            self.objective = MSELoss(self.datarc['num_speakers'], self.modelrc["mask_type"])
        elif self.modelrc["loss_type"] == "SISDR":
            self.objective = SISDRLoss(self.datarc['num_speakers'], 
                    n_fft=self.datarc['n_fft'], 
                    hop_length=self.datarc['hop_length'], 
                    win_length=self.datarc['win_length'], 
                    window=self.datarc['window'], 
                    center=self.datarc['center'])
        else:
            raise ValueError("Loss type not defined.")
        
        self.best_score = -10000

    def _get_train_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.loaderrc["train_batchsize"],
            shuffle=True,
            num_workers=self.loaderrc["num_workers"],
            drop_last=False,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.loaderrc["eval_batchsize"],
            shuffle=False,
            num_workers=self.loaderrc["num_workers"],
            drop_last=False,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )

    def get_train_dataloader(self):
        return self._get_train_dataloader(self.train_dataset)

    def get_dev_dataloader(self):
        return self._get_eval_dataloader(self.dev_dataset)

    def get_test_dataloader(self):
        return self._get_eval_dataloader(self.test_dataset)

    def forward(self, features, uttname_list, source_attr, source_wav, target_attr, target_wav_list, feat_length, wav_length, 
            records=None, logger=None, prefix=None, global_step=0, **kwargs):
        """
        Args:
            features:
                list of unpadded features [feat1, feat2, ...]
                each feat is in torch.FloatTensor and already
                put in the device assigned by command-line args

            uttname_list:
                list of utterance names

            source_attr:
                source_attr is a dict containing the STFT information 
                for the mixture. source_attr['magnitude'] stores the STFT
                magnitude, source_attr['phase'] stores the STFT phase and
                source_attr['stft'] stores the raw STFT feature. The shape
                is [bs, max_length, feat_dim]

            source_wav:
                source_wav contains the raw waveform for the mixture,
                and it has the shape of [bs, max_wav_length]

            target_attr:
                similar to source_attr, it contains the STFT information
                for individual sources. It only has two keys ('magnitude' and 'phase')
                target_attr['magnitude'] is a list of length n_srcs, and
                target_attr['magnitude'][i] has the shape [bs, max_length, feat_dim]

            target_wav_list:
                target_wav_list contains the raw waveform for the individual
                sources, and it is a list of length n_srcs. target_wav_list[0]
                has the shape [bs, max_wav_length]

            feat_length:
                length of STFT features

            wav_length:
                length of raw waveform

        Return:
            loss:
                the loss to be optimized, should not be detached
        """
        
        # match the feature length to STFT feature length
        features = match_length(features, feat_length)
        features = pack_sequence(features)
        mask = self.model(features)

        # evaluate the separation quality of predict sources
        if not self.training:
            predict_stfts = [torch.squeeze(m * source_attr['stft'].to(device)) for m in mask]
            predict_stfts_np = [np.transpose(s.data.cpu().numpy()) for s in predict_stfts]

            assert len(wav_length) == 1
            # reconstruct the signal using iSTFT
            predict_srcs_np = [librosa.istft(stft_mat, 
                hop_length=self.datarc['hop_length'], 
                win_length=self.datarc['win_length'], 
                window=self.datarc['window'], 
                center=self.datarc['center'],
                length=wav_length[0]) for stft_mat in predict_stfts_np]
            predict_srcs_np = np.stack(predict_srcs_np, 0)
            gt_srcs_np = torch.cat(target_wav_list, 0).data.cpu().numpy()
            mix_np = source_wav.data.cpu().numpy()

            utt_metrics = get_metrics(
                mix_np,
                gt_srcs_np,
                predict_srcs_np,
                sample_rate = self.datarc['rate'],
                metrics_list = COMPUTE_METRICS,
                compute_permutation=True,
            )

            for metric in COMPUTE_METRICS:
                input_metric = "input_" + metric
                assert metric in utt_metrics and input_metric in utt_metrics
                imp = utt_metrics[metric] - utt_metrics[input_metric]
                if metric not in records:
                    records[metric] = []
                records[metric].append(imp)

        if self.loss_type == "MSE": # mean square loss
            loss = self.objective.compute_loss(mask, feat_length, source_attr, target_attr)
        elif self.loss_type == "SISDR": # end-to-end SI-SNR loss
            loss = self.loss_func.compute_loss(masks, input_sizes, source_attr, wav_length, target_wav_list)
        else:
            raise ValueError("Loss type not defined.")
        return loss

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
        if self.training:
            return 0
        else:
            for metric in COMPUTE_METRICS:
                avg_metric = np.mean(records[metric])
                records[metric] = avg_metric

                logger.add_scalar(
                    f'{prefix}'+metric,
                    records[metric],
                    global_step=global_step
                )

            save_ckpt = []
            assert 'si_sdr' in records
            if records['si_sdr'] > self.best_score:
                self.best_score = records['si_sdr']
                save_ckpt.append("modelbest.ckpt")

            return save_ckpt