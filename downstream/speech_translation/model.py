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

    def __init__(self, vocab_size, padding_idx, **config):
        super().__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(vocab_size, config['d_model'], padding_idx=padding_idx)
        self.pos_encoding = PositionalEncoding(config['d_model'])
        self.model = nn.Transformer(**config)
        self.d_model = config['d_model']

        self.apply(self.init_weights)

    def forward(self, features, features_length, decoder_input_idx):
        # feature: (T, B, *), decoder_input_idx: (T, B)
        # mask features
        # features_key_padding_mask = (features[:, :, 0] == self.padding_idx).transpose(0, 1)
        features_key_padding_mask = self.create_mask_with_lengths(features_length, features.size(0)).to(features.device)
        tgt_key_padding_mask = (decoder_input_idx == self.padding_idx).transpose(0, 1)
        decoder_input = self.pos_encoding(self.embedding(decoder_input_idx))
        tgt_mask = self.model.generate_square_subsequent_mask(decoder_input_idx.size(0)).to(features.device)
        logit = self.model(
            features, 
            decoder_input,
            tgt_mask = tgt_mask,
            src_key_padding_mask = features_key_padding_mask,
            tgt_key_padding_mask = tgt_key_padding_mask,
            memory_key_padding_mask = features_key_padding_mask,
        )
        predict = logit @ self.embedding.weight.transpose(0, 1)
        return predict

    def incremental_decode(self, features, features_length, start_ids, max_len):

        tgt = self.embedding(start_ids)
        tgt = self.pos_encoding(tgt)
        # features_key_padding_mask = (features[:, :, 0] == self.padding_idx).transpose(0, 1)
        features_key_padding_mask = self.create_mask_with_lengths(features_length, features.size(0)).to(features.device)

        memory = self.model.encoder(
            features,
            src_key_padding_mask=features_key_padding_mask,
        )

        tgt_full_mask = self.model.generate_square_subsequent_mask(max_len).to(features.device)
        output_ids = start_ids
        for i in range(max_len):
            tgt_mask = tgt_full_mask[:i+1, :i+1]
            ## no adding tgt mask
            decoder_out = self.model.decoder(
                tgt,
                memory,
                tgt_mask = tgt_mask,
                memory_key_padding_mask = features_key_padding_mask,
            )
            logit = decoder_out @ self.embedding.weight.data.transpose(0, 1)
            step_out_ids = torch.max(logit[-1], -1).indices.unsqueeze(0)
            step_out_emb = self.pos_encoding(self.embedding(step_out_ids))
            output_ids = torch.cat((output_ids, step_out_ids))
            tgt = torch.cat((tgt, step_out_emb))

        return output_ids

    def create_mask_with_lengths(self, lengths, max_length):
        mask = torch.full((len(lengths), max_length), True)
        for i in range(len(lengths)):
            mask[i, :lengths[i]] = False
        return mask

    def init_weights(self, module):

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.d_model**0.5)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * self.d_model ** 0.5 + self.pe[:x.size(0), :]
        return self.dropout(x)

class Seq2SeqRNN(nn.Module):

    def __init__(self, vocab_size, padding_idx, d_model, n_layers, bidirectional=True):

        super().__init__()

        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.n_layers = n_layers

        if bidirectional:
            assert d_model % 2 == 0
            d_encoder = d_model // 2
        else:
            d_encoder = d_model

        self.encoder = nn.LSTM(
            input_size = d_model,
            hidden_size = d_encoder,
            num_layers = n_layers,
            bidirectional = bidirectional,
        )

        self.decoder = nn.LSTM(
            input_size = d_model,
            hidden_size = d_model,
            num_layers = n_layers,
        )

    def forward(self, features, features_length, decoder_input_idx):
        
        _, state = self.encoder(features)

        state = self.transform_state(state)

        decoder_input_emb = self.embedding(decoder_input_idx)
        logit, _ = self.decoder(decoder_input_emb, state)
        predict = logit @ self.embedding.weight.transpose(0, 1)
        return predict

    def transform_state(self, state):

        h, c = state

        h = h.view(self.n_layers, -1, h.size(1), h.size(2))
        h = torch.cat([h[:, i, :, :] for i in range(h.size(1))], dim=-1)

        c = c.view(self.n_layers, -1, c.size(1), c.size(2))
        c = torch.cat([c[:, i, :, :] for i in range(c.size(1))], dim=-1)

        return (h, c)

    def incremental_decode(self, features, features_length, start_ids, max_len):

        _, state = self.encoder(features)
        state = self.transform_state(state)
        output_ids = start_ids
        for i in range(max_len):
            decoder_input_emb = self.embedding(output_ids[-1]).unsqueeze(0)
            logit, state = self.decoder(decoder_input_emb, state)
            predict = logit @ self.embedding.weight.transpose(0, 1)
            predict_id = torch.argmax(predict, dim=-1)
            output_ids = torch.cat((output_ids, predict_id), dim=0)
        return output_ids


# class ToyRNNS2S(nn.Module):

#     def __init__(self, vocab_size, padding_idx, hidden_size, num_encoder_layers, num_decoder_layers):
#         super().__init__()

#         self.embed = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)

#         self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_encoder_layers, bidirectional=True)

#         self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_decoder_layers)

#     def forward(self, features, decoder_input_idx):

#         embed = self.embed(decoder_input_idx)

#         _, state = self.encoder(features)


# class RNNSeq2Seq(nn.Module):

#     def __init__(self, vocab_size, padding_idx, hidden_size, num_encoder_layers, num_decoder_layers):
#         super().__init__()

#         self.embed = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)

#         if num_encoder_layers <= 0:
#             self.encoder = None
#         else:
#             self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_encoder_layers, bidirectional=True)

#         self.decoder = AttentionalRNNDecoder(hidden_size, num_decoder_layers)

#     def forward(self, features, decoder_input_idx):
        
#         if self.encoder is not None:
#             features = self.encoder(features)

#         embed = self.embed(decoder_input_idx)
#         logit = self.decoder(features, embed)

#         logit = logit @ self.embed.weight.data.transpose(0, 1)

#         return logit
    
#     def incremental_decode(self, feature, start_ids):

# class AttentionalRNNDecoder(nn.Module):

#     def __init__(self, hidden_size, num_layers):

#         super().__init__()
        
#         self.rnns = nn.ModuleList(
#             [AttentionalRNNDecoderLayer(hidden_size) for _ in range(num_layers)]
#         )

#     def forward(self, memory, in_features, state=None):

#         outputs = []

#         for i in range(in_features.size(0)):
#             output = in_features[i].unsqueeze(0)
#             for layers in self.rnns:
#                 output, state = layers(memory, output, state)
#             outputs.append(output)

#         return torch.cat(outputs, dim=0)



# class AttentionalRNNDecoderLayer(nn.Module):

#     def __init__(self, hidden_size):
#         super().__init__()

#         self.rnn = nn.LSTM(hidden_size, hidden_size)
#         self.multi_attn = nn.MultiheadAttention(hidden_size, 4)

#     def forward(self, features, input, state):

#         output, state = self.rnn(input, state)
#         attn_out, _ = self.multi_attn(output, features, features)

#         return attn_out, state