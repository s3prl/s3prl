
###############
# IMPORTATION #
###############
import torch.nn as nn
#-------------#
from fairseq.models.wav2vec.wav2vec2 import TransformerEncoder, TransformerSentenceEncoderLayer

#######
# APC #
#######
args = {
    "input_feat": 80,
    "encoder_embed_dim": 768,
    "encoder_layers": 12,
    "dropout": 0.1,
    "activation_dropout": 0,
    "dropout_input": 0.1,
    "attention_dropout": 0.1,
    "encoder_layerdrop": 0.05,
    "conv_pos": 128,
    "conv_pos_groups": 16,
    "encoder_ffn_embed_dim": 3072,
    "encoder_attention_heads": 12,
    "activation_fn": "gelu",
    "layer_norm_first": False,
}

class Config:
    pass

class Decoar2(nn.Module):
    def __init__(self):
        """
            input_size: an int indicating the input feature size, e.g., 80 for Mel.
            hidden_size: an int indicating the RNN hidden size.
            num_layers: an int indicating the number of RNN layers.
            dropout: a float indicating the RNN dropout rate.
            residual: a bool indicating whether to apply residual connections.
        """
        super(Decoar2, self).__init__()
        config = Config()
        for arg_name, arg_val in args.items():
            setattr(config, arg_name, arg_val)

        self.post_extract_proj = nn.Linear(config.input_feat, config.encoder_embed_dim)
        self.dropout_input = nn.Dropout(config.dropout_input)
        self.encoder = TransformerEncoder(config)

    def forward(self, features, padding_mask=None):
        features = self.post_extract_proj(features)

        x = self.dropout_input(features)

        if padding_mask is not None:
            extra = padding_mask.size(1) % features.size(1)
            if extra > 0:
                padding_mask = padding_mask[:, :-extra]
            padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
            padding_mask = padding_mask.all(-1)

        x, layer_results = self.encoder(x, padding_mask=padding_mask, layer=None)

        return x, layer_results