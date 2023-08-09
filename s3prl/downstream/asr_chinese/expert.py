# Copyright (c) Facebook, Inc. All Rights Reserved

import os
import math  # noqa
import editdistance
from pathlib import Path
from argparse import Namespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence

from .model import *
from ..model import *
from .dataset import SequenceDataset
from .dictionary import Dictionary


def token_to_word(text):
    # Hard coding but it is only used here for now.
    # Assumption that units are characters. Doesn't handle BPE.
    # Inter-character separator is " " and inter-word separator is "|".
    return text.replace(" ", "").replace("|", " ").strip()


def get_decoder(decoder_args_dict, dictionary):
    decoder_args = Namespace(**decoder_args_dict)

    if decoder_args.decoder_type == "kenlm":
        from .w2l_decoder import W2lKenLMDecoder
        decoder_args.beam_size_token = len(dictionary)
        if isinstance(decoder_args.unk_weight, str):
            decoder_args.unk_weight = eval(decoder_args.unk_weight)
        return W2lKenLMDecoder(decoder_args, dictionary)

    return None


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
        self.upstream_rate = upstream_rate
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']
        self.expdir = expdir

        self.dictionary = Dictionary.load(self.datarc.get("dict_path", str(Path(__file__).parent / "char.dict")))
    
        self.projector = nn.Linear(upstream_dim, self.modelrc['project_dim'])
        model_cls = eval(self.modelrc['select'])
        model_conf = self.modelrc[self.modelrc['select']]
        self.model = model_cls(
            self.modelrc['project_dim'],
            len(self.dictionary.symbols),
            upstream_rate,
            **model_conf,
        )
        self.blank = self.dictionary.bos()
        self.objective = nn.CTCLoss(
            blank=self.blank,
            zero_infinity=self.datarc['zero_infinity']
        )
        decoder_args = self.datarc.get('decoder_args')
        self.decoder = get_decoder(decoder_args, self.dictionary)
        self.register_buffer('best_score', torch.ones(1) * 100)

    # Interface
    def get_dataloader(self, split):
        """
        Args:
            split: string
                The name of the dataloader, can be train/dev/test-clean/test-other for asr

        Return:
            a torch.utils.data.DataLoader returning each batch in the format of:

            [wav1, wav2, ...], your_other_contents1, your_other_contents2, ...

            where wav1, wav2 ... are in variable length
            each wav is torch.FloatTensor in cpu with:
                1. dim() == 1
                2. sample_rate == 16000
                3. directly loaded by torchaudio
        """
        if not hasattr(self, f'{split}_dataset'):
            batch_size = self.datarc['batch_size'] if split == "train" else self.datarc['eval_batch_size']
            setattr(self, f'{split}_dataset', SequenceDataset(split, batch_size, self.dictionary, **self.datarc))

        if split == 'train':
            return self._get_train_dataloader(self.train_dataset)
        else:
            return self._get_eval_dataloader(getattr(self, f'{split}_dataset'))

    def _get_train_dataloader(self, dataset):
        sampler = DistributedSampler(dataset) if is_initialized() else None
        return DataLoader(
            dataset, batch_size=1,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn,
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset, batch_size=1,
            shuffle=False, num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn
        )

    def _compute_metrics(self, pred_tokens_all, pred_words_all, target_tokens_all, target_words_all):
        """Computes WER and UER given the prediction and true transcriptions"""
        unit_error_sum = 0.0
        word_error_sum = 0.0
        unit_length_sum = 0
        word_length_sum = 0

        for pred_tokens, pred_words, target_tokens, target_words in zip(
            pred_tokens_all, pred_words_all, target_tokens_all, target_words_all):

            pred_tokens = pred_tokens.split()
            target_tokens = target_tokens.split()
            unit_error_sum += editdistance.eval(pred_tokens, target_tokens)
            unit_length_sum += len(target_tokens)

            word_error_sum += editdistance.eval(pred_words, target_words)
            word_length_sum += len(target_words)

        uer, wer = 100.0, 100.0
        if unit_length_sum > 0:
            uer = 100.0 * unit_error_sum / unit_length_sum
        if word_length_sum > 0:
            wer = 100.0 * word_error_sum / word_length_sum

        return uer, wer

    def _decode(self, log_probs, input_lens):
        """Decoder that take log probabilities as input and outputs decoded seq"""
        pred_tokens_batch = []
        pred_words_batch = []

        for log_prob, in_len in zip(log_probs, input_lens):
            log_prob = log_prob[:in_len].unsqueeze(0)
            decoded = None
            if self.decoder is not None and not self.training:
                decoded = self.decoder.decode(log_prob)
                if len(decoded) >= 1:
                    decoded = decoded[0]
                    decoded = None if len(decoded) < 1 else decoded[0]
            
            pred_token_ids = log_prob.argmax(dim=-1).unique_consecutive()
            pred_token_ids = pred_token_ids[pred_token_ids != self.blank].tolist()
            pred_tokens = self.dictionary.string(pred_token_ids)

            if decoded is not None and "words" in decoded:
                pred_words = decoded["words"]
            else:
                pred_words = token_to_word(pred_tokens).split()

            pred_tokens_batch.append(pred_tokens)
            pred_words_batch.append(pred_words)

        return pred_tokens_batch, pred_words_batch

    def _get_log_probs(self, features):
        device = features[0].device
        features_len = torch.IntTensor([len(feat) for feat in features])
        features = pad_sequence(features, batch_first=True).to(device=device)
        features = self.projector(features)
        logits, log_probs_len = self.model(features, features_len)
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        return log_probs, log_probs_len

    def inference(self, features, filenames):
        log_probs, log_probs_len = self._get_log_probs(features)
        _, pred_words_batch = self._decode(log_probs.float().contiguous().cpu(), log_probs_len)
        hyps = [' '.join(hyp) for hyp in pred_words_batch]

        if filenames != []:
            with open(Path(self.expdir) / "inference.ark", "w") as file:
                for hyp, filename in zip(hyps, filenames):
                    file.write(f"{filename} {hyp}\n")

        return hyps

    # Interface
    def forward(self, split, features, labels, filenames, records, **kwargs):
        """
        Args:
            split: string
                The name of the dataloader, can be train/dev/test-clean/test-other for asr

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
        log_probs, log_probs_len = self._get_log_probs(features)
        device = features[0].device
        labels = [torch.IntTensor(l) for l in labels]
        labels_len = torch.IntTensor([len(label) for label in labels]).to(device=device)
        labels = pad_sequence(
            labels,
            batch_first=True,
            padding_value=self.dictionary.pad(),
        ).to(device=device)

        loss = self.objective(
                log_probs.transpose(0, 1), # (N, T, C) -> (T, N, C)
                labels,
                log_probs_len,
                labels_len,
            )
        records['loss'].append(loss.item())

        target_tokens_batch = []
        target_words_batch = []
        for label in labels:
            label_idx = (label != self.dictionary.pad()) & (
                label != self.dictionary.eos()
            )
            target_token_ids = label[label_idx].tolist()
            target_tokens = self.dictionary.string(target_token_ids)
            target_words = token_to_word(target_tokens).split()

            target_tokens_batch.append(target_tokens)
            target_words_batch.append(target_words)

        with torch.no_grad():
            pred_tokens_batch, pred_words_batch = self._decode(log_probs.float().contiguous().cpu(), log_probs_len)

        records['target_tokens'] += target_tokens_batch
        records['target_words'] += target_words_batch
        records['pred_tokens'] += pred_tokens_batch
        records['pred_words'] += pred_words_batch
        records['filenames'] += filenames

        return loss

    # interface
    def log_records(self, split, records, logger, global_step, batch_ids, total_batch_num, **kwargs):
        """
        Args:
            split: string
                'train':
                    records and batchids contain contents for `log_step` batches
                    `log_step` is defined in your downstream config
                    eg. downstream/example/config.yaml

                'dev' or 'test-clean' or 'test-other' :
                    records and batchids contain contents for the entire evaluation dataset

            records:
                defaultdict(list), contents already prepared by self.forward

            logger:
                Tensorboard SummaryWriter
                please use f'{your_task_name}/{split}-{key}' as key name to log your contents,
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
        loss = torch.FloatTensor(records['loss']).mean().item()
        print(f'{split} loss: {loss}')

        uer, wer = self._compute_metrics(
            records['pred_tokens'],
            records['pred_words'],
            records['target_tokens'],
            records['target_words'],
        )

        logger.add_scalar(f'asr/{split}-loss', loss, global_step=global_step)
        logger.add_scalar(f'asr/{split}-uer', uer, global_step=global_step)
        logger.add_scalar(f'asr/{split}-wer', wer, global_step=global_step)
        print(f'{split} uer: {uer}')
        print(f'{split} wer: {wer}')

        save_names = []
        if split == 'dev-clean' and wer < self.best_score:
            self.best_score = torch.ones(1) * wer
            save_names.append(f'{split}-best.ckpt')

        if 'test' in split or 'dev' in split:
            lm = "noLM" if self.decoder is None else "LM"
            hyp_ark = open(os.path.join(self.expdir, f'{split}-{lm}-hyp.ark'), 'w')
            ref_ark = open(os.path.join(self.expdir, f'{split}-{lm}-ref.ark'), 'w')
            for filename, hyp, ref in zip(records['filenames'], records['pred_words'], records['target_words']):
                hyp = ' '.join(hyp)
                ref = ' '.join(ref)
                hyp_ark.write(f'{filename} {hyp}\n')
                ref_ark.write(f'{filename} {ref}\n')
            hyp_ark.close()
            ref_ark.close()

        return save_names
