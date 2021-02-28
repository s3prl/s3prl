import os
import math
import torch
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from ..asr.model import *
from .text import load_text_encoder
from .data import load_dataset
from .utility import cal_er


class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, upstream_rate, downstream_expert, expdir, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.corpus = downstream_expert['corpus']

        # Text tokenizer
        self.tokenizer = load_text_encoder(**downstream_expert['text'])

        modelrc = downstream_expert['model']
        self.projector = nn.Linear(upstream_dim, modelrc['project_dim'])

        model_select = downstream_expert['model']['select']
        self.model = eval(model_select)(
            modelrc['project_dim'],
            self.tokenizer.vocab_size,
            upstream_rate,
            **modelrc[model_select],
        )
        self.objective = nn.CTCLoss(
            blank = self.tokenizer.pad_idx,
            zero_infinity = modelrc['zero_infinity'],
        )
        self.register_buffer('best_score', torch.ones(1) * 100)

    # Interface
    def get_dataloader(self, split):
        return load_dataset(split, self.tokenizer, self.corpus)

    # Interface
    def forward(self, split, features, labels, filenames, records, **kwargs):
        device = features[0].device
        features_len = torch.IntTensor([len(feat) for feat in features])
        labels_len = torch.IntTensor([len(label) for label in labels])
        features = pad_sequence(features, batch_first=True)
        labels = pad_sequence(
            labels,
            batch_first=True,
            padding_value=self.tokenizer.pad_idx,
        ).to(device=device)

        features = self.projector(features)
        log_probs, log_probs_len = self.model(features, features_len)

        loss = self.objective(
                log_probs.transpose(0, 1), # (N, T, C) -> (T, N, C)
                labels,
                log_probs_len,
                labels_len,
            )
        wer, hypothesis, groundtruth = cal_er(self.tokenizer, log_probs, labels, ctc=True)

        records['loss'].append(loss.item())
        records['wer'].append(wer)

        # store text for the first sample in a batch
        records['hypothesis'].append(hypothesis[0])
        records['groundtruth'].append(groundtruth[0])
        records['filename'].append(filenames[0])

        return loss

    # interface
    def log_records(self, split, records, logger, global_step, **kwargs):
        save_names = []
        for key, values in records.items():
            if key in ['loss', 'wer']:
                average = torch.FloatTensor(values).mean().item()
                logger.add_scalar(
                    f'slot_filling/{split}-{key}',
                    average,
                    global_step=global_step
                )
                if key == 'wer':
                    if split == 'dev' and average < self.best_score:
                        self.best_score = torch.ones(1) * average
                        save_names.append(f'{split}-best.ckpt')

        for i in range(0, len(records['filename']), round(len(records['filename']) / 5)):
            filename = records['filename'][i]
            hypothesis = records['hypothesis'][i]
            groundtruth = records['groundtruth'][i]
            logger.add_text(
                f'slot_filling/{split}-{filename}',
                f'**hypothesis**: {hypothesis}<br>**groundtruth**: {groundtruth}',
                global_step=global_step,
            )

        return save_names
