import torch
import torch.nn as nn
import math

class Model(nn.Module):
    def __init__(self, input_dim, output_class_num, **kwargs):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_dim, output_class_num)

    def forward(self, features):
        pooled = features.mean(dim=1)
        predicted = self.linear(pooled)
        return predicted

class Transformer(nn.Module):

    def __init__(self, vocab_size, padding_idx=None, **config):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config['d_model'], padding_idx=padding_idx)
        self.pos_encoding = PositionalEncoding(config['d_model'])
        self.model = nn.Transformer(**config)

    def forward(self, features, decoder_input_idx):
        # feature: (T, B, *), decoder_input_idx: (T, B)
        decoder_input = self.pos_encoding(self.embedding(decoder_input_idx))
        tgt_mask = self.model.generate_square_subsequent_mask(decoder_input_idx.size(0)).to(features.device)
        logit = self.model(features, decoder_input, tgt_mask=tgt_mask)
        predict = logit @ self.embedding.weight.transpose(0, 1)
        return predict

    def incremental_decode(self, features, start_ids, max_len):

        tgt = self.embedding(start_ids)
        tgt = self.pos_encoding(tgt)
        memory = self.model.encoder(features)
        tgt_full_mask = self.model.generate_square_subsequent_mask(max_len).to(features.device)
        output_ids = start_ids
        for i in range(max_len):
            tgt_mask = tgt_full_mask[:i+1, :i+1]
            decoder_out = self.model.decoder(tgt, memory, tgt_mask = tgt_mask)
            logit = decoder_out @ self.embedding.weight.data.transpose(0, 1)
            step_out_ids = torch.max(logit[-1], -1).indices.unsqueeze(0)
            step_out_emb = self.pos_encoding(self.embedding(step_out_ids))
            output_ids = torch.cat((output_ids, step_out_ids))
            tgt = torch.cat((tgt, step_out_emb))

        return output_ids

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)