import os
import math
import torch
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer
# from .model import Model
from .dataset import AtisDataset
import pandas as pd
from collections import Counter
import wandb
from .model import AttenDecoderModel, Seq2SeqTransformer, generate_square_subsequent_mask, create_mask, greedy_decode
from .metric import parse_entity, entity_f1_score, parse_BI_entity, parse_BIO_entity, uer

PAD_IDX = 0
EOS_IDX = 1

class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, downstream_expert, **kwargs):
        super(DownstreamExpert, self).__init__()
        torch.backends.cudnn.enabled = False
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']
        self.seq_loss = torch.nn.CrossEntropyLoss(ignore_index=0, label_smoothing=self.modelrc['label_smoothing'])
        
        self.base_path = self.datarc['file_path']   
        self.is_BI = self.datarc['is_BI']
        self.is_BIO = self.datarc['is_BIO']
        self.unit_path = self.datarc['unit_path'] if 'unit_path' in self.datarc else None
        self.unit_tokenizer_path = self.datarc['unit_tokenizer_path'] if 'unit_tokenizer_path' in self.datarc else None
        self.unit_size = self.datarc['unit_size'] + 3 if 'unit_size' in self.datarc else None
        self.unit_tokenizer = None

        if self.unit_tokenizer_path is not None: 
            self.unit_tokenizer = Tokenizer.from_file(self.unit_tokenizer_path)


        aug_config = downstream_expert['augmentation'] if 'augmentation' in downstream_expert else None
        if self.is_BI: 
            self.tokenizer = Tokenizer.from_file(os.path.join(self.base_path, 'BI_tokenizer.json'))
            self.train_dataset = AtisDataset(os.path.join(self.base_path, 'sv_BI_train.csv'), os.path.join(self.base_path, 'train'), self.tokenizer, aug_config, unit_path=self.unit_path, unit_tokenizer=self.unit_tokenizer)
            self.dev_dataset = AtisDataset(os.path.join(self.base_path, 'sv_BI_dev.csv'), os.path.join(self.base_path, 'dev'), self.tokenizer, unit_path=self.unit_path, unit_tokenizer=self.unit_tokenizer)
            self.test_dataset = AtisDataset(os.path.join(self.base_path, 'sv_BI_test.csv'),os.path.join(self.base_path, 'test'), self.tokenizer, unit_path=self.unit_path, unit_tokenizer=self.unit_tokenizer)
        elif self.is_BIO: 
            self.tokenizer = Tokenizer.from_file(os.path.join(self.base_path, 'BIO_tokenizer.json'))
            self.train_dataset = AtisDataset(os.path.join(self.base_path, 'sv_BIO_train.csv'), os.path.join(self.base_path, 'train'), self.tokenizer, aug_config, unit_path=self.unit_path, unit_tokenizer=self.unit_tokenizer)
            self.dev_dataset = AtisDataset(os.path.join(self.base_path, 'sv_BIO_dev.csv'), os.path.join(self.base_path, 'dev'), self.tokenizer, unit_path=self.unit_path, unit_tokenizer=self.unit_tokenizer)
            self.test_dataset = AtisDataset(os.path.join(self.base_path, 'sv_BIO_test.csv'),os.path.join(self.base_path, 'test'), self.tokenizer, unit_path=self.unit_path, unit_tokenizer=self.unit_tokenizer)
        else:
            self.tokenizer = Tokenizer.from_file(os.path.join(self.base_path, 'tokenizer.json'))
            self.train_dataset = AtisDataset(os.path.join(self.base_path, 'atis_sv_train.csv'), os.path.join(self.base_path, 'train'), self.tokenizer, aug_config, unit_path=self.unit_path, unit_tokenizer=self.unit_tokenizer)
            self.dev_dataset = AtisDataset(os.path.join(self.base_path, 'atis_sv_dev.csv'), os.path.join(self.base_path, 'dev'), self.tokenizer, unit_path=self.unit_path, unit_tokenizer=self.unit_tokenizer)
            self.test_dataset = AtisDataset(os.path.join(self.base_path, 'atis_sv_test.csv'),os.path.join(self.base_path, 'test'), self.tokenizer, unit_path=self.unit_path, unit_tokenizer=self.unit_tokenizer)
        
        self.connector = nn.Linear(upstream_dim, self.modelrc['input_dim'])
        self.vocab_size = self.modelrc['input_dim']
        self.ctc_weight = self.modelrc['ctc_weight']
        self.is_transformer = self.modelrc['is_transformer']
        
        self.num_encoder_layers = self.modelrc['num_encoder_layers']
        self.num_decoder_layers = self.modelrc['num_decoder_layers']
        self.emb_size = self.modelrc['emb_size']
        self.nhead = self.modelrc['nhead']
        self.is_dual_decoder = self.modelrc['is_dual_decoder']
        self.unit_decode_weight = self.modelrc['unit_decode_weight']

        self.is_unit = False
        if self.unit_path is not None and self.ctc_weight != 0.:
            self.is_unit = True
            self.ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=False)
        
        if self.is_transformer: 
            self.model = Seq2SeqTransformer(num_encoder_layers=self.modelrc['num_encoder_layers'], 
                                            num_decoder_layers=self.modelrc['num_decoder_layers'],
                                            emb_size=self.modelrc['emb_size'], 
                                            nhead=self.modelrc['nhead'], 
                                            tgt_vocab_size=self.modelrc['vocab_size'],
                                            dim_feedforward=self.modelrc['dim_feedforward'],
                                            is_unit=self.is_unit, 
                                            unit_size=self.unit_size,
                                            is_dual_decoder=self.is_dual_decoder,
                                            )
        else:
            self.model = AttenDecoderModel(self.modelrc['input_dim'], self.vocab_size)

        

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

    def get_train_dataloader(self):
        return self._get_train_dataloader(self.train_dataset)

    def get_dev_dataloader(self):
        return self._get_eval_dataloader(self.dev_dataset)

    def get_test_dataloader(self):
        return self._get_eval_dataloader(self.test_dataset)

    def get_dataloader(self, mode):
        return eval(f'self.get_{mode}_dataloader')()

    # Interface
    def forward(self, mode, features, labels, units=None, records=None, **kwargs):
        features_pad = pad_sequence(features, batch_first=True)
        DEVICE = features_pad.device

        if units is not None: 
            units = [torch.LongTensor(unit).to(DEVICE) for unit in units]
            units_pad = pad_sequence(units, batch_first=True)
            encode_len = torch.IntTensor([feature.shape[0] for feature in features]).to(DEVICE)
            unit_len = torch.IntTensor([len(u) for u in units])
            unit_input = units_pad[:, :-1]
            unit_out = units_pad[:, 1:]
        
        labels = [torch.LongTensor(label).to(DEVICE) for label in labels]
        label_len = [len(l) for l in labels]

        labels_pad = pad_sequence(labels, batch_first=True)
        tgt_input = labels_pad[:, :-1]
        tgt_out = labels_pad[:, 1:]
        
        attention_mask = [torch.ones((feature.shape[0])).to(DEVICE) for feature in features] 
        attention_mask_pad = pad_sequence(attention_mask,batch_first=True)
        attention_mask_pad = (1.0 - attention_mask_pad) * -100000.0

        src_mask, tgt_mask, tgt_padding_mask = create_mask(features_pad, tgt_input)
        # for unit_mask
        _ , unit_mask, unit_padding_mask = create_mask(features_pad, unit_input)

        features_pad = self.connector(features_pad)

        # train mode 
        if mode == 'train':
            if self.is_transformer: 
                att_output, ctc_output, unit_output = self.model(features_pad, tgt_input, src_mask, tgt_mask, attention_mask_pad, tgt_padding_mask, attention_mask_pad, unit_input, self.ctc_weight, unit_mask)
            else:
                ctc_output, encode_len, att_output, att_seq, dec_state = self.model(features_pad, max(label_len), 0.7, labels_pad)
        # dev, test mode
        else: 
            if self.is_transformer: 
                if self.ctc_weight < 1.0:
                    ys = list(greedy_decode(self.model, features_pad, src_mask, max_len=30).flatten().cpu().numpy().astype(int))
                    ys = ys[1:]
                att_output, ctc_output, unit_output = self.model(features_pad, tgt_input, src_mask, tgt_mask, attention_mask_pad, tgt_padding_mask, attention_mask_pad, unit_input, self.ctc_weight, unit_mask)
            else:
                ctc_output, encode_len, att_output, att_seq, dec_state = self.model(features_pad, max(label_len))
        
        total_loss = 0
    
        if ctc_output is not None and self.ctc_weight > 0.0:
            # T, N, C for ctc_loss input
            ctc_loss = self.ctc_loss(ctc_output.transpose(0,1), units_pad, encode_len, unit_len)
            total_loss += ctc_loss * self.ctc_weight
            records['ctc_loss'].append(ctc_loss.cpu().item())

            log_probs = nn.functional.log_softmax(ctc_output, dim=-1)
            pred_tokens = log_probs.argmax(dim=-1)
            
            # decode for ctc
            filtered_tokens = []
            for pred_token in pred_tokens:
                filtered_tokens.append(self.ctc_decode(pred_token.tolist(), ignore_repeat=True))

            groundtruth = [self.ctc_decode(g.tolist()) for g in units]
            # record uer for token
            UER = uer(filtered_tokens, groundtruth)
            records['UER'] += [UER]
            if mode != 'train':
                print(filtered_tokens, groundtruth)
                print(UER)

        if unit_output is not None: 
            b,t,_ = unit_output.shape
            unit_decode_loss = self.seq_loss(unit_output.reshape(b*t, -1), unit_out.reshape(-1))
            total_loss += unit_decode_loss * self.unit_decode_weight
            records['unit_decode_loss'].append(unit_decode_loss.cpu().item())

        if att_output is not None and self.ctc_weight < 1.0: 
            b,t,_ = att_output.shape
            att_loss = self.seq_loss(att_output.reshape(b*t, -1), tgt_out.reshape(-1))
            total_loss += att_loss * (1 - self.ctc_weight)
            records['att_loss'].append(att_loss.cpu().item())

            hyps = att_output.argmax(dim=-1).detach().tolist()
            gts = tgt_out.detach().tolist()

            f1s = []
            f1s_ys = []
            accs = []
            accs_ys = []
            
            for hyp, gt in zip(hyps, gts):
                intent_gt, intent_hyp = None, None
                if self.is_BI:
                    d_gt = parse_BI_entity(gt, self.tokenizer)
                    d_hyp = parse_BI_entity(hyp, self.tokenizer)
                elif self.is_BIO:
                    d_gt, intent_gt = parse_BIO_entity(gt, self.tokenizer)
                    d_hyp, intent_hyp = parse_BIO_entity(hyp, self.tokenizer)
                else:
                    d_gt = parse_entity(gt)
                    d_hyp = parse_entity(hyp)
                
                if intent_gt is not None: 
                    if intent_gt == intent_hyp:
                        acc = 1.0
                    else: 
                        acc = 0.0
                    accs.append(acc)

                f1 = entity_f1_score(d_gt, d_hyp)
                if mode != 'train':
                    if self.is_BI:
                        d_ys = parse_BI_entity(ys, self.tokenizer)
                    elif self.is_BIO:
                        d_ys, intent_ys = parse_BIO_entity(ys, self.tokenizer)
                    else:
                        d_ys = parse_entity(ys)

                    if intent_ys is not None: 
                        if intent_gt == intent_hyp:
                            acc = 1.0
                        else: 
                            acc = 0.0
                        accs_ys.append(acc)

                    f1_ys = entity_f1_score(d_gt, d_ys)
                    f1s_ys.append(f1_ys)
                    print(d_gt, d_hyp, d_ys)
                    print(f1, f1_ys)
                    print(gt, hyp, ys)
                    print(intent_gt, intent_hyp, intent_ys)
                    
                f1s.append(f1)

            if mode != 'train':
                records['f1_greedy'] += f1s_ys
                
            records['f1'] += f1s
            if self.is_BIO:
                records['intent_acc'] += accs

        records['tot_loss'].append(total_loss.cpu().item())
        return total_loss

    # interface
    def log_records(self, mode, records, logger, global_step, **kwargs):
        for key, values in records.items():
            average = torch.FloatTensor(values).mean().item()
            wandb.log({f'atis/{mode}-{key}': average})
    
    def ctc_decode(self, idxs, ignore_repeat=False):
        vocabs = []
        for t, idx in enumerate(idxs):
            v = chr(idx)
            if idx == PAD_IDX or (ignore_repeat and t > 0 and idx == idxs[t-1]):
                continue
            elif idx == EOS_IDX:
                break
            else:
                vocabs.append(v)
        return "".join(vocabs)


