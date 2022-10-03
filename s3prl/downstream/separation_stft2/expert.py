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
from pathlib import Path
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
from .loss import SepLoss, SISDRLoss
from itertools import permutations

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

COMPUTE_METRICS = ["si_sdr"]
EPS = 1e-10

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

    def __init__(self, upstream_dim, upstream_rate, downstream_expert, expdir, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.upstream_rate = upstream_rate
        self.datarc = downstream_expert["datarc"]
        self.loaderrc = downstream_expert["loaderrc"]
        self.modelrc = downstream_expert["modelrc"]
        self.expdir = expdir

        self.train_dataset = SeparationDataset(
                data_dir=self.loaderrc["train_dir"],
                rate=self.datarc['rate'],
                src=self.datarc['src'],
                tgt=self.datarc['tgt'],
                n_fft=self.datarc['n_fft'],
                hop_length=self.upstream_rate,
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
                hop_length=self.upstream_rate,
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
                hop_length=self.upstream_rate,
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
        self.log = self.modelrc["log"]
        self.objective = SepLoss(self.datarc['num_speakers'], self.loss_type, self.modelrc["mask_type"], self.log)
        
        self.register_buffer("best_score", torch.ones(1) * -10000)

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
        if mode == "train":
            return self._get_train_dataloader(self.train_dataset)
        elif mode == "dev":
            return self._get_eval_dataloader(self.dev_dataset)
        elif mode == "test":
            return self._get_eval_dataloader(self.test_dataset)

    def forward(self, mode, features, uttname_list, source_attr, source_wav, target_attr, target_wav_list, feat_length, wav_length, records, **kwargs):
        """
        Args:
            mode: string
                'train', 'dev' or 'test' for this forward step

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

            records:
                defaultdict(list), by appending contents into records,
                these contents can be averaged and logged on Tensorboard
                later by self.log_records every log_step

        Return:
            loss:
                the loss to be optimized, should not be detached
        """
        # match the feature length to STFT feature length
        features = match_length(features, feat_length)
        features = pack_sequence(features)
        mask_list = self.model(features)

        # evaluate the separation quality of predict sources
        if mode == 'dev' or mode == 'test':
            if self.log == 'none':
                predict_stfts = [torch.squeeze(m.cpu() * source_attr['stft']) for m in mask_list]
                predict_stfts_np = [np.transpose(s.data.numpy()) for s in predict_stfts]
            elif self.log == 'log1p':
                phase = source_attr['stft'] / (source_attr['stft'].abs() + EPS)
                predict_stfts = [torch.squeeze(torch.expm1(m.cpu() * torch.log1p(source_attr['stft'].abs())) * phase) for m in mask_list]
                predict_stfts_np = [np.transpose(s.data.numpy()) for s in predict_stfts]
            else:
                raise ValueError("log type not defined.")

            assert len(wav_length) == 1
            # reconstruct the signal using iSTFT
            predict_srcs_np = [librosa.util.fix_length(librosa.istft(stft_mat,
                hop_length=self.upstream_rate,
                win_length=self.datarc['win_length'], 
                window=self.datarc['window'], 
                center=self.datarc['center']), size=wav_length[0]) for stft_mat in predict_stfts_np]
            predict_srcs_np = np.stack(predict_srcs_np, 0)
            gt_srcs_np = torch.cat(target_wav_list, 0).data.cpu().numpy()
            mix_np = source_wav.data.cpu().numpy()

            perm_list = [list(perm) for perm in list(permutations(range(len(gt_srcs_np))))]
            utt_metrics_list = [get_metrics(
                mix_np,
                gt_srcs_np,
                predict_srcs_np[perm, :],
                sample_rate = self.datarc['rate'],
                metrics_list = COMPUTE_METRICS,
                compute_permutation=False,
            ) for perm in perm_list]

            utt_metrics = {}
            for metric in COMPUTE_METRICS:
                input_metric = "input_" + metric
                utt_metrics[input_metric] = utt_metrics_list[0][input_metric]
                utt_metrics[metric] = np.max([k[metric] for k in utt_metrics_list])

            for metric in COMPUTE_METRICS:
                input_metric = "input_" + metric
                assert metric in utt_metrics and input_metric in utt_metrics
                imp = utt_metrics[metric] - utt_metrics[input_metric]
                if metric not in records:
                    records[metric] = []
                if metric == "si_sdr":
                    records[metric].append(imp)
                elif metric == "stoi" or metric == "pesq":
                    records[metric].append(utt_metrics[metric])
                else:
                    raise ValueError("Metric type not defined.")

            assert 'batch_id' in kwargs
            if kwargs['batch_id'] % 1000 == 0: # Save the prediction every 1000 examples
                records['mix'].append(mix_np)
                records['hypo'].append(predict_srcs_np)
                records['ref'].append(gt_srcs_np)
                records['uttname'].append(uttname_list[0])

        if self.loss_type == "MSE" or self.loss_type == "L1":
            loss = self.objective.compute_loss(mask_list, feat_length, source_attr, target_attr)
        else:
            raise ValueError("Loss type not defined.")

        records["loss"].append(loss.item())
        return loss

    # interface
    def log_records(
        self, mode, records, logger, global_step, batch_ids, total_batch_num, **kwargs
    ):
        """
        Args:
            mode: string
                'train':
                    records and batchids contain contents for `log_step` batches
                    `log_step` is defined in your downstream config
                    eg. downstream/example/config.yaml
                'dev' or 'test' :
                    records and batchids contain contents for the entire evaluation dataset

            records:
                defaultdict(list), contents already appended

            logger:
                Tensorboard SummaryWriter
                please use f'{prefix}your_content_name' as key name
                to log your customized contents

            global_step:
                The global_step when training, which is helpful for Tensorboard logging

            batch_ids:
                The batches contained in records when enumerating over the dataloader

            total_batch_num:
                The total amount of batches in the dataloader

        Return:
            a list of string
                Each string is a filename we wish to use to save the current model
                according to the evaluation result, like the best.ckpt on the dev set
                You can return nothing or an empty list when no need to save the checkpoint
        """
        if mode == 'train':
            avg_loss = np.mean(records["loss"])
            logger.add_scalar(
                f"separation_stft2/{mode}-loss", avg_loss, global_step=global_step
            )
            return []
        else:
            avg_loss = np.mean(records["loss"])
            logger.add_scalar(
                f"separation_stft2/{mode}-loss", avg_loss, global_step=global_step
            )
            with (Path(self.expdir) / f"{mode}_metrics.txt").open("w") as output:
                for metric in COMPUTE_METRICS:
                    avg_metric = np.mean(records[metric])
                    if mode == "test" or mode == "dev":
                        print("Average {} of {} utts: {:.4f}".format(metric, len(records[metric]), avg_metric))
                        print(metric, avg_metric, file=output)

                    logger.add_scalar(
                        f'separation_stft2/{mode}-'+metric,
                        avg_metric,
                        global_step=global_step
                    )

            save_ckpt = []
            assert 'si_sdr' in records
            if mode == "dev" and np.mean(records['si_sdr']) > self.best_score:
                self.best_score = torch.ones(1) * np.mean(records['si_sdr'])
                save_ckpt.append(f"best-states-{mode}.ckpt")

            for s in ['mix', 'ref', 'hypo', 'uttname']:
                assert s in records
            for i in range(len(records['uttname'])):
                utt = records['uttname'][i]
                mix_wav = records['mix'][i][0, :]
                mix_wav = librosa.util.normalize(mix_wav, norm=np.inf, axis=None)
                logger.add_audio('step{:06d}_{}_mix.wav'.format(global_step, utt), mix_wav, global_step=global_step, sample_rate=self.datarc['rate'])

                for j in range(records['ref'][i].shape[0]):
                    ref_wav = records['ref'][i][j, :]
                    hypo_wav = records['hypo'][i][j, :]
                    ref_wav = librosa.util.normalize(ref_wav, norm=np.inf, axis=None)
                    hypo_wav = librosa.util.normalize(hypo_wav, norm=np.inf, axis=None)
                    logger.add_audio('step{:06d}_{}_ref_s{}.wav'.format(global_step, utt, j+1), ref_wav, global_step=global_step, sample_rate=self.datarc['rate'])
                    logger.add_audio('step{:06d}_{}_hypo_s{}.wav'.format(global_step, utt, j+1), hypo_wav, global_step=global_step, sample_rate=self.datarc['rate'])
            return save_ckpt
