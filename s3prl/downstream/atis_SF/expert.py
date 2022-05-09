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
from .model import AttenDecoderModel, Seq2SeqTransformer, generate_square_subsequent_mask, create_mask

class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, downstream_expert, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']
        self.seq_loss = torch.nn.CrossEntropyLoss(ignore_index=0)
        # self.get_dataset()
        self.base_path = self.datarc['file_path']   
        self.tokenizer = Tokenizer.from_file(os.path.join(self.base_path, 'tokenizer.json'))
        aug_config = downstream_expert['augmentation'] if 'augmentation' in downstream_expert else None

        self.train_dataset = AtisDataset(os.path.join(self.base_path, 'atis_sv_train.csv'), os.path.join(self.base_path, 'train'), self.tokenizer, aug_config)
        self.dev_dataset = AtisDataset(os.path.join(self.base_path, 'atis_sv_dev.csv'), os.path.join(self.base_path, 'dev'), self.tokenizer)
        self.test_dataset = AtisDataset(os.path.join(self.base_path, 'atis_sv_test.csv'),os.path.join(self.base_path, 'test'), self.tokenizer)
        
        self.connector = nn.Linear(upstream_dim, self.modelrc['input_dim'])
        self.objective = nn.CrossEntropyLoss()
        self.vocab_size = self.modelrc['input_dim']
        self.enable_ctc = self.modelrc['enable_ctc']
        self.is_transformer = self.modelrc['is_transformer']
        self.num_encoder_layers = self.modelrc['num_encoder_layers']
        self.num_decoder_layers = self.modelrc['num_decoder_layers']
        self.emb_size = self.modelrc['emb_size']
        self.nhead = self.modelrc['nhead']
        

        if self.is_transformer: 
            self.model = Seq2SeqTransformer(num_encoder_layers=self.modelrc['num_encoder_layers'], 
                                            num_decoder_layers=self.modelrc['num_decoder_layers'],
                                            emb_size=self.modelrc['emb_size'], 
                                            nhead=self.modelrc['nhead'], 
                                            tgt_vocab_size=self.modelrc['vocab_size'])
        else:
            self.model = AttenDecoderModel(self.modelrc['input_dim'], self.vocab_size)

        if self.enable_ctc:
            self.ctc_layer = nn.Sequential(
                nn.Linear(self.emb_size, self.vocab_size), 
                nn.ReLU() 
            )

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
    def forward(self, mode, features, labels, records=None, **kwargs):
        features_pad = pad_sequence(features, batch_first=True)
        DEVICE = features_pad.device
        labels = [torch.LongTensor(label).to(DEVICE) for label in labels]
        label_len = [len(l) for l in labels]

        labels_pad = pad_sequence(labels, batch_first=True)
        tgt_input = labels_pad[:, :-1]
        tgt_out = labels_pad[:, 1:]
        
        attention_mask = [torch.ones((feature.shape[0])).to(DEVICE) for feature in features] 
        attention_mask_pad = pad_sequence(attention_mask,batch_first=True)
        attention_mask_pad = (1.0 - attention_mask_pad) * -100000.0

        src_mask, tgt_mask, tgt_padding_mask = create_mask(features_pad, tgt_input)

        features_pad = self.connector(features_pad)

        # train mode 
        if mode == 'train':
            if self.is_transformer: 
                
                att_output = self.model(features_pad, tgt_input, src_mask, tgt_mask, attention_mask_pad, tgt_padding_mask, attention_mask_pad)
            else:
                ctc_output, encode_len, att_output, att_seq, dec_state = self.model(features_pad, max(label_len), 0.7, labels_pad)
        # dev, test mode
        else: 
            if self.is_transformer: 
                att_output = self.model(features_pad, tgt_input, src_mask, tgt_mask, attention_mask_pad, tgt_padding_mask, attention_mask_pad)
            else:
                ctc_output, encode_len, att_output, att_seq, dec_state = self.model(features_pad, max(label_len))
        total_loss = 0
        # intent_logits = self.model(features_pad, attention_mask_pad.cuda())
        # if ctc_output is not None:
        #     if self.paras.cudnn_ctc:
        #         ctc_loss = self.ctc_loss(ctc_output.transpose(0,1), 
        #                                     txt.to_sparse().values().to(device='cpu',dtype=torch.int32),
        #                                     [ctc_output.shape[1]]*len(ctc_output),
        #                                     #[int(encode_len.max()) for _ in encode_len],
        #                                     txt_len.cpu().tolist())
        #     else:
        #         ctc_loss = self.ctc_loss(ctc_output.transpose(0,1), txt, encode_len, txt_len)
        #     total_loss += ctc_loss*self.model.ctc_weight
        #     del encode_len

        if att_output is not None:
            b,t,_ = att_output.shape
            att_loss = self.seq_loss(att_output.reshape(b*t,-1),tgt_out.reshape(-1))
            total_loss += att_loss
      
        records['loss'] = total_loss.cpu().item()
        return total_loss

    # interface
    def log_records(self, mode, records, logger, global_step, **kwargs):
        for key, values in records.items():
            wandb.log({f'atis/{mode}-{key}': values})

