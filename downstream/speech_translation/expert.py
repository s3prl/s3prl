import os
import math
import torch
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence

from .model import Model, Transformer
from .dataset import COVOST2Dataset

import sentencepiece
import sacrebleu
from tqdm import tqdm

class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, upstream_rate, downstream_expert, expdir, **kwargs):
        """
        Args:
            upstream_dim: int
                Different upstream will give different representation dimension
                You might want to first project them to the same dimension

            upstream_rate: int
                160: for upstream with 10 ms per frame
                320: for upstream with 20 ms per frame
            
            downstream_expert: dict
                The 'downstream_expert' field specified in your downstream config file
                eg. downstream/example/config.yaml

            expdir: string
                The expdir from command-line argument, you should save all results into
                this directory, like some logging files.

            **kwargs: dict
                All the arguments specified by the argparser in run_downstream.py
                and all the other fields in config.yaml, in case you need it.
                
                Note1. Feel free to add new argument for __init__ as long as it is
                a command-line argument or a config field. You can check the constructor
                code in downstream/runner.py
        """

        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.max_length = downstream_expert['max_length']
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']
        self.tokenizerrc = downstream_expert['tokenizerrc']

        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=self.tokenizerrc['model_path'])
        self.tokenizer.set_encode_extra_options("bos:eos")

        self.dataset = {}
        for split in ['train', 'dev', 'test']:
            self.dataset[split] = COVOST2Dataset(
                self.datarc['src_lang'],
                self.datarc['tgt_lang'],
                split, 
                self.datarc['root_dir'],
                self.datarc['tsv_dir'],
                self.tokenizer, 
                self.max_length
            )
        
        self.connector = nn.Linear(upstream_dim, self.modelrc['d_model'])
        # self.connector = nn.Linear(upstream_dim, self.modelrc['hidden_size'])
        
        self.model = Transformer(
            vocab_size = self.tokenizer.vocab_size(),
            padding_idx = self.tokenizer.pad_id(),
            **self.modelrc,
        )

        # self.model = RNNSeq2Seq(
        #     vocab_size = self.tokenizer.vocab_size(),
        #     padding_idx = self.tokenizer.pad_id(),
        #     **self.modelrc,
        # )
        
        self.objective = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id())
        self.register_buffer('best_score', torch.zeros(1))

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
            return self._get_train_dataloader(self.dataset['train'])
        else: 
            return self._get_eval_dataloader(self.dataset[mode])


    def _get_train_dataloader(self, dataset):
        sampler = DistributedSampler(dataset) if is_initialized() else None
        return DataLoader(
            dataset, batch_size=self.datarc['train_batch_size'],
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn
        )


    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset, batch_size=self.datarc['eval_batch_size'],
            shuffle=False, num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn
        )


    # Interface
    def forward(self, mode, features, your_other_contents1, records, **kwargs):
        """
        Args:
            mode: string
                'train', 'dev' or 'test' for this forward step

            features:
                list of unpadded features [feat1, feat2, ...]
                each feat is in torch.FloatTensor and already
                put in the device assigned by command-line args

            your_other_contents1, ... :
                in the order defined by your dataloader (dataset + collate_fn)
                these are all in cpu, and you can move them to the same device
                as features

            records:
                defaultdict(list), by appending contents into records,
                these contents can be averaged and logged on Tensorboard
                later by self.log_records (also customized by you)

                Note1. downstream/runner.py will call self.log_records
                    1. every `log_step` during training
                    2. once after evalute the whole dev/test dataloader

                Note2. `log_step` is defined in your downstream config
                eg. downstream/example/config.yaml

        Return:
            loss:
                the loss to be optimized, should not be detached
                a single scalar in torch.FloatTensor
        """
        features = pad_sequence(features, batch_first=False, padding_value=self.tokenizer.pad_id())
        features = self.connector(features)

        utterance_labels = your_other_contents1
        labels = pad_sequence(utterance_labels, batch_first=False, padding_value=self.tokenizer.pad_id())
        labels = labels.to(features.device)

        # tqdm.write(f"features size: {features.size()}, labels size: {labels.size()}")
        if mode == 'train' or mode == 'dev':
            predicted = self.model(features, labels)
            shifted_labels = torch.cat((labels[1:], torch.full((1, labels.size(1)), self.tokenizer.pad_id()).to(predicted.device)), dim=0)
            loss = self.objective(predicted.view(-1, predicted.size(-1)), shifted_labels.view(-1))
            records['loss'].append(loss.item())
            predicted_classid = predicted.max(dim=-1).indices
        elif mode == 'test':
            start_ids = torch.full((1, features.size(1)), self.tokenizer.bos_id(), device = features.device)
            predicted_classid = self.model.incremental_decode(features, start_ids, self.max_length)
            loss = None
        else:
            raise
    
        
        # predict_classid: (N, B) 
        predicted_classid = predicted_classid.cpu().transpose(0, 1).tolist()
        
        ### remove tokens behind <eos>
        for index, predict in enumerate(predicted_classid):
            if self.tokenizer.eos_id() in predict:
                eos_pos = predict.index(self.tokenizer.eos_id())
                predicted_classid[index] = predicted_classid[index][:eos_pos+1]
        
        predict_text = self.tokenizer.decode(predicted_classid)
        records['pred'] += predict_text
        records['refs'] +=  self.tokenizer.decode([l.tolist() for l in utterance_labels])

        return loss


    # interface
    def log_records(self, mode, records, logger, global_step, batch_ids, total_batch_num, **kwargs):
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
                defaultdict(list), contents already prepared by self.forward

            logger:
                Tensorboard SummaryWriter
                please use f'{your_task_name}/{mode}-{key}' as key name to log your contents,
                preventing conflict with the logging of other tasks

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
        save_names = []
        for key, values in records.items():
            if key == 'refs' or key == 'pred': continue
            average = torch.FloatTensor(values).mean().item()
            logger.add_scalar(
                f'example/{mode}-{key}',
                average,
                global_step=global_step
            )
        
        if mode == 'dev' or mode == 'test':
            bleu = sacrebleu.corpus_bleu(records['pred'], [records['refs']])
            if bleu.score > self.best_score:
                self.best_score = torch.ones(1) * bleu.score
                save_names.append(f'{mode}-best.ckpt')
            
            logger.add_scalar(
                f'example/{mode}-bleu',
                bleu.score,
                global_step=global_step
            )
            tqdm.write(f"[{mode}]:bleu-{bleu.score}")
            tqdm.write(f"pred: {records['pred'][-1]}")
            tqdm.write(f"refs: {records['refs'][-1]}")
        return save_names
