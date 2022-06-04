# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/mockingjay/builder.py ]
#   Synopsis     [ build the transformer model for downstream usage ]
#   Author       [ Andy T. Liu (https://github.com/andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


import copy
import math
import random

###############
# IMPORTATION #
###############
import sys
from distutils.util import strtobool
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import yaml
from torch.nn.utils.rnn import pad_sequence

import s3prl.optimizers

from ..baseline.extracter import get_extracter
from ..baseline.preprocessor import get_preprocessor
from .model import TransformerConfig, TransformerModel, TransformerSpecPredictionHead


#######################
# TRANSFORMER BUILDER #
#######################
class TransformerBuilder(nn.Module):
    """
    A builder class for all pre-trained Transformer.
    Child classes only need to implement the __init__() and forward() method.
    """

    def __init__(
        self, options, inp_dim=-1, config=None, on_the_fly_config=None, verbose=False
    ):
        super(TransformerBuilder, self).__init__()

        # read config
        if config is not None:
            self.config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
        else:
            # Since some old checkpoints contained pickled scheduler which needs 'optimizers'
            # module which is now moved into s3prl package.
            original_optimizer = sys.modules.get("optimizers")
            sys.modules["optimizers"] = s3prl.optimizers

            self.all_states = torch.load(options["ckpt_file"], map_location="cpu")
            if "transformer" in self.all_states["Config"]:
                # for legacy ckpts, 'Config' stored the upstream config, and runner config was stored under 'Runner'.
                self.config = self.all_states["Config"]
            elif "transformer" in self.all_states["Upstream_Config"]:
                # For new ckpts, 'Upstream_Config' is used to store the upstream config, and runner config is now stored under 'Config'.
                self.config = self.all_states["Upstream_Config"]
            else:
                raise NotImplementedError

            del sys.modules["optimizers"]
            if original_optimizer is not None:
                sys.modules["optimizers"] = original_optimizer

        # parse the options dict
        self.load = bool(strtobool(options["load_pretrain"]))
        self.no_grad = bool(strtobool(options["no_grad"]))
        self.spec_aug = bool(strtobool(options["spec_aug"]))
        self.spec_aug_prev = bool(strtobool(options["spec_aug_prev"]))
        self.output_hidden_states = bool(strtobool(options["output_hidden_states"]))
        self.select_layer = int(options["select_layer"])
        self.permute_input = bool(strtobool(options["permute_input"]))
        if (not self.no_grad) and (not self.spec_aug_prev):
            raise RuntimeError("Only one of them can be set False!")
        if str(options["dropout"]) != "default":  # increase dropout if specified
            self.config["transformer"]["hidden_dropout_prob"] = float(
                options["dropout"]
            )
            self.config["transformer"]["attention_probs_dropout_prob"] = float(
                options["dropout"]
            )

        # Set model config
        self.model_config = TransformerConfig(self.config["transformer"])
        self.hidden_size = self.model_config.hidden_size
        self.num_layers = self.model_config.num_hidden_layers
        self.max_input_length = self.config["task"]["sequence_length"]

        if on_the_fly_config is not None:
            self.config["audio"] = yaml.load(
                open(on_the_fly_config, "r"), Loader=yaml.FullLoader
            )
        if "audio" in self.config:
            if "kaldi" in self.config["audio"]:
                self.extracter, self.inp_dim, _ = get_extracter(self.config["audio"])
                self.spec_dim = self.inp_dim
            else:
                self.extracter, self.inp_dim, self.spec_dim = get_preprocessor(
                    self.config["audio"], process_input_only=True
                )
                self.target_level = self.config["audio"]["target_level"]
        elif inp_dim != -1:
            self.extracter, self.inp_dim, self.spec_dim = None, inp_dim, inp_dim
        else:
            self.extracter, self.inp_dim, self.spec_dim = (
                None,
                self.config["transformer"]["input_dim"],
                self.config["transformer"]["input_dim"],
            )

        if self.max_input_length > 0 and verbose:
            print("[Transformer] - Maximum input length: ", self.max_input_length)
        if not (self.select_layer in list(range(-1, self.num_layers))):
            raise RuntimeError("Out of range int for 'select_layer'!")

    def _normalize_wav_decibel(self, wav):
        """Normalize the signal to the target level"""
        rms = wav.pow(2).mean().pow(0.5)
        scalar = (10 ** (self.target_level / 20)) / (rms + 1e-10)
        wav = wav * scalar
        return wav

    def load_model(self, transformer_model, state_dict, verbose=False):
        try:
            old_keys = []
            new_keys = []
            for key in state_dict.keys():
                new_key = None
                if "gamma" in key:
                    new_key = key.replace("gamma", "weight")
                if "beta" in key:
                    new_key = key.replace("beta", "bias")
                if new_key:
                    old_keys.append(key)
                    new_keys.append(new_key)
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

            missing_keys = []
            unexpected_keys = []
            error_msgs = []
            # copy state_dict so _load_from_state_dict can modify it
            metadata = getattr(state_dict, "_metadata", None)
            state_dict = state_dict.copy()
            if metadata is not None:
                state_dict._metadata = metadata

            def load(module, prefix=""):
                local_metadata = (
                    {} if metadata is None else metadata.get(prefix[:-1], {})
                )
                module._load_from_state_dict(
                    state_dict,
                    prefix,
                    local_metadata,
                    True,
                    missing_keys,
                    unexpected_keys,
                    error_msgs,
                )
                for name, child in module._modules.items():
                    if child is not None:
                        load(child, prefix + name + ".")

            load(transformer_model)
            if len(missing_keys) > 0:
                print(
                    "Weights of {} not initialized from pretrained model: {}".format(
                        transformer_model.__class__.__name__, missing_keys
                    )
                )
            if len(unexpected_keys) > 0:
                print(
                    "Weights from pretrained model not used in {}: {}".format(
                        transformer_model.__class__.__name__, unexpected_keys
                    )
                )
            if len(error_msgs) > 0:
                raise RuntimeError(
                    "Error(s) in loading state_dict for {}:\n\t{}".format(
                        transformer_model.__class__.__name__, "\n\t".join(error_msgs)
                    )
                )
            if verbose:
                print("[Transformer] - Pre-trained weights loaded!")
            return transformer_model

        except:
            raise RuntimeError("[Transformer] - Pre-trained weights NOT loaded!")

    def process_input_data(self, feat):
        """Process input data for the model"""

        # add arbitary batch axis B if input `feat` has shape of TxD
        if len(feat.shape) == 2:
            feat = feat.unsqueeze(0)
        # input `feat` should have shape BxTxD
        elif len(feat.shape) != 3:
            raise ValueError(
                "Input argument `feat` has invalid shape: {}".format(feat.shape)
            )

        # Record length for each uttr
        spec_len = (feat.sum(dim=-1) != 0).long().sum(dim=-1)
        spec_len = spec_len.tolist()

        batch_size = feat.shape[0]
        seq_len = feat.shape[1]

        pos_enc = position_encoding(seq_len, self.hidden_size)  # (seq_len, hidden_size)
        attn_mask = np.ones((batch_size, seq_len))  # (batch_size, seq_len)

        # zero vectors for padding dimension
        for idx in range(len(feat)):
            attn_mask[idx][spec_len[idx] :] = 0

        if (
            self.spec_aug
            and self.spec_aug_prev
            and self.model.training
            and self.inp_dim > 1
        ):
            feat = spec_augment(
                feat, mask_T=70, mask_F=9, num_T=2, num_F=2, p=1.0
            )  # (batch_size, seq_len, feature_dim * dr)
        feat = feat.to(dtype=torch.float32)  # (batch_size, seq_len, feature_dim * dr)
        pos_enc = (
            torch.FloatTensor(pos_enc)
            .to(device=feat.device, dtype=torch.float32)
            .expand(feat.size(0), *pos_enc.size())
        )  # (batch_size, seq_len, hidden_size)
        attn_mask = torch.FloatTensor(attn_mask).to(
            device=feat.device, dtype=torch.float32
        )  # (batch_size, seq_len)
        return feat, pos_enc, attn_mask  # (x, pos_enc, attention_mask)

    def _forward(self, x):

        if self.permute_input:
            x = x.permute(1, 0, 2).contiguous()  # (T, B, D) -> (B, T, D)
        input_len = x.shape[1]

        # forward the whole sequence at once
        if self.max_input_length == 0 or input_len <= self.max_input_length:
            feat, pos_enc, attn_mask = self.process_input_data(x)  # x shape: (B, T, D)
            x = self.model(
                feat,
                pos_enc,
                attn_mask,
                output_all_encoded_layers=self.output_hidden_states
                or self.select_layer != -1,
            )
            x = (
                torch.stack(x) if isinstance(x, list) else x
            )  # (B, T, D) or # (N, B, T, D)
        # forward the sequence in chunks then concat
        else:
            chunks = torch.chunk(
                x, chunks=math.ceil(input_len / self.max_input_length), dim=1
            )
            x_ = []
            for chunk in chunks:
                feat, pos_enc, attn_mask = self.process_input_data(
                    chunk
                )  # x shape: (B, T, D)
                chunk = self.model(
                    feat,
                    pos_enc,
                    attn_mask,
                    output_all_encoded_layers=self.output_hidden_states
                    or self.select_layer != -1,
                )  # (B, T, D) or # (N, B, T, D)
                x_.append(torch.stack(chunk) if type(chunk) is list else chunk)
            x = torch.cat(
                x_,
                dim=2 if (self.output_hidden_states or self.select_layer != -1) else 1,
            )
        # x can be a single hidden state or a list of hidden states

        if self.output_hidden_states:
            hidden_states = x
            x = hidden_states[self.select_layer]

        if (
            self.spec_aug
            and not self.spec_aug_prev
            and self.model.training
            and self.inp_dim > 1
        ):
            x = spec_augment(
                x, mask_T=70, mask_F=86, num_T=2, num_F=2, p=1.0
            )  # (B, T, D)

        # permute to output
        if self.permute_input:
            x = x.permute(1, 0, 2).contiguous()  # (B, T, D) -> (T, B, D)
            if self.output_hidden_states:
                # (N, B, T, D) -> (N, T, B, D)
                hidden_states = hidden_states.transpose(1, 2).contiguous()

        if self.output_hidden_states:
            return x, hidden_states
        return x  # (B, T, D) or (T, B, D)


##########################
# PRETRAINED TRANSFORMER #
##########################
class PretrainedTransformer(TransformerBuilder):
    """
    Use this class to extract features from the Transformer model,
    or to finetune the pre-trained Transformer with any downstream tasks.
    Also, this class is `pytorch-kaldi` ready,
    hence we need to use `str` instead of `bool` in the options dict,
    as pytorch-kaldi scripts will pass in str.

    Params:
        `options`: a python dictionary containing the following keys:
            ckpt_file: str, a path specifying the pre-trained ckpt file
            load_pretrain: str, ['True', 'False'], whether to load pre-trained weights
            no_grad: str, ['True', 'False'], whether to use torch.no_grad over forward, this determines should torch build the computational graph
            dropout: float/str, use float to modify dropout value during downstream finetune, or use the str `default` for pre-train default values
            spec_aug: str, ['True', 'False'], whether to apply the SpecAugment technique
            spec_aug_prev: str, ['True', 'False'], True: apply spec augment on input (i.e. acoustic features); False: apply on output (i.e. the hidden states)
            weighted_sum: str, ['True', 'False'], whether to use a learnable weighted sum to integrate hidden representations from all layers, if False then use the one specified in `select_layer`
            select_layer: int, select from all hidden representations, set to -1 to select the last (will only be used when weighted_sum is False)
            permute_input: str, ['True', 'False'], this attribute is for the forward method. Ture: input / ouput of shape (T, B, D); False: input / ouput of shape (B, T, D)
        `intput_dim`: int, input dimension of model
        `config`: optional, reads the given yaml config and not use the config stored in `ckpt_file`

    An example `options` dictionary:
    options = {
        'ckpt_file'     : './result/result_transformer/libri_sd1337_fmllrBase960-F-N-K-RA/states-1000000.ckpt',
        'load_pretrain' : 'True',
        'no_grad'       : 'True',
        'dropout'       : 'default',
        'spec_aug'      : 'False',
        'spec_aug_prev' : 'True',
        'weighted_sum'  : 'False',
        'select_layer'  : -1,
        'permute_input' : 'False',
    }
    """

    def __init__(
        self, options, inp_dim, config=None, online_config=None, verbose=False
    ):
        super(PretrainedTransformer, self).__init__(
            options, inp_dim, config, online_config, verbose
        )

        # Build model
        self.model = TransformerModel(self.model_config, self.inp_dim)
        self.model.eval() if self.no_grad else self.model.train()
        self.out_dim = (
            self.hidden_size
        )  # This attribute is necessary, for pytorch-kaldi and run_downstream.py

        # Load from a PyTorch state_dict
        if self.load:
            self.model = self.load_model(
                self.model, self.all_states["Transformer"], verbose
            )
            if verbose:
                print(
                    "[Transformer] - Number of parameters: "
                    + str(
                        sum(
                            p.numel()
                            for p in self.model.parameters()
                            if p.requires_grad
                        )
                    )
                )

    def forward(self, x):
        if self.extracter is not None:
            if "kaldi" in self.config["audio"]:
                x = [self.extracter(x_i) for x_i in x]
                x = pad_sequence(x, batch_first=True)
            else:
                x = [self._normalize_wav_decibel(x_i) for x_i in x]
                x_lens = [len(x_) for x_ in x]
                x = pad_sequence(x, batch_first=True)
                x = x.unsqueeze(
                    1
                )  # (batch_size, audio_len) -> (batch_size, 1, audio_len)
                x = self.extracter(x, wavs_len=x_lens)[0]
        if self.no_grad:
            with torch.no_grad():
                x = self._forward(x)
        else:
            x = self._forward(x)
        return x


####################################
# PRETRAINED TRANSFORMER WITH HEAD #
####################################
class PretrainedTransformerWithHead(PretrainedTransformer):
    def __init__(
        self, options, inp_dim, config=None, online_config=None, verbose=False
    ):
        super(PretrainedTransformerWithHead, self).__init__(
            options, inp_dim, config, online_config, verbose
        )

        # build head
        self.SpecHead = TransformerSpecPredictionHead(self.model_config, self.spec_dim)
        self.SpecHead.eval() if self.no_grad else self.SpecHead.train()

        # Load from a PyTorch state_dict
        if self.load:
            self.SpecHead.load_state_dict(self.all_states["SpecHead"])
            if verbose:
                print(
                    "[Spec Transformer] - Number of parameters: "
                    + str(
                        sum(
                            p.numel()
                            for p in self.SpecHead.parameters()
                            if p.requires_grad
                        )
                    )
                )

    def forward(self, x):
        if self.extracter is not None:
            if "kaldi" in self.config["audio"]:
                x = [self.extracter(x_i) for x_i in x]
                x = pad_sequence(x, batch_first=True)
            else:
                x = [self._normalize_wav_decibel(x_i) for x_i in x]
                x_lens = [len(x_) for x_ in x]
                x = pad_sequence(x, batch_first=True)
                x = x.unsqueeze(
                    1
                )  # (batch_size, audio_len) -> (batch_size, 1, audio_len)
                x = self.extracter(x, wavs_len=x_lens)[0]
        if self.no_grad:
            with torch.no_grad():
                x = self._forward(x)
                x, _ = self.SpecHead(x)
        else:
            x = self._forward(x)
            x, _ = self.SpecHead(x)
        return x


#######################
# POSITIONAL ENCODING #
#######################
MAX_SEQLEN = 24000


@lru_cache(maxsize=128)
def get_sinusoid_table(hidden_size):
    def _cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / hidden_size)

    def _get_posi_angle_vec(position):
        return [_cal_angle(position, hid_j) for hid_j in range(hidden_size)]

    sinusoid_table = np.array(
        [_get_posi_angle_vec(pos_i) for pos_i in range(MAX_SEQLEN)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)


def position_encoding(seq_len, hidden_size):
    """position encoding table"""
    table = get_sinusoid_table(hidden_size)[:seq_len]
    # no extra CPU and GPU memory allocation
    # after getting the (seq_len, hidden_size) tensor, one should first put
    # this tensor into GPU then expand it
    return table  # (seq_len, hidden_size)


################
# SPEC AUGMENT #
################
"""
Process training data for the supervised ASR model by 
masking to time-steps and channels during training
which delays overfitting and significantly improves the final accuracy numbers.
Input:
    `spec`: input real frames, with shape: (batch_size, seq_len, feature_dim)
    `mask_T`: the time mask parameter T described in the SpecAugment paper, 
              we use default values based on the LD Policy
              (In paper: T=100, we use 70 since we are training on the 100 hr subset only)
    `mask_F`: the frequency mask parameter F described in the SpecAugment paper, 
              we use default values based on the LD Policy
              (In paper: F=27:D=80*3 -> F=9:D=80, where D is acoustic dimension)
    `num_T` : the number of time masks applied (In paper: mT=2)
    `num_F` : the number of frequency masks applied (In paper: mF=2)
    `p` : upper bound ratio (In paper: p=1.0)
Output:
    `spec`: augmented frames, with shape: (batch_size, seq_len, feature_dim)
"""


def spec_augment(spec, mask_T=70, mask_F=9, num_T=2, num_F=2, p=1.0):
    def _start_to_intervals(starts, consecutive):
        tiled = starts.expand(consecutive, starts.size(0)).permute(1, 0)
        offset = torch.arange(consecutive).expand_as(tiled)
        intervals = tiled + offset
        return intervals.view(-1)

    with torch.no_grad():
        upper_bound = (
            spec.shape[1] * p
        )  # upper bound on the time mask so that a time mask cannot be wider than p times the number of time steps

        for idx in range(spec.shape[0]):

            # time masking
            if mask_T > 0 and mask_T < upper_bound:
                for _ in range(num_T):
                    rand_consecutive = random.randint(0, mask_T)
                    chosen_start = torch.randperm(spec.shape[1] - rand_consecutive)[:1]
                    chosen_intervals = _start_to_intervals(
                        chosen_start, rand_consecutive
                    )
                    spec[idx, chosen_intervals, :] = 0

            # frequency masking
            if mask_F > 0:
                for _ in range(num_F):
                    rand_bandwidth = random.randint(0, mask_F)
                    chosen_start = torch.randperm(spec.shape[2] - rand_bandwidth)[:1]
                    chosen_intervals = _start_to_intervals(chosen_start, rand_bandwidth)
                    spec[idx, :, chosen_intervals] = 0

        return spec


#######
# LIN #
#######
"""
Linear Input Networks (LIN) for domain adaptation
Params:
    `options`: a python dictionary containing arguments for pytorch kaldi, give None if not using with pytorch-kaldi:
    `intput_dim`: int, input dimension of model
"""


class LIN(nn.Module):
    def __init__(self, options, inp_dim):
        super(LIN, self).__init__()

        self.out_dim = inp_dim  # This attribute is for pytorch-kaldi
        self.linear = nn.Linear(inp_dim, inp_dim)
        self.linear.weight.data.copy_(torch.eye(inp_dim))

        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.linear = self.linear.to(self.device)
        self.linear.train()

    def forward(self, x):
        x = self.linear(x)
        return x
