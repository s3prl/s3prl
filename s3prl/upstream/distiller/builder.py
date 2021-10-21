"""
    Builder for Distiller
    Author: Heng-Jui Chang (https://github.com/vectominist)
"""

import sys
import copy
import math
from distutils.util import strtobool
import yaml
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from .model import DistillerConfig, DistillerModel
import s3prl.optimizers


class DistillerBuilder(nn.Module):
    """
    A builder class for all pre-trained Distiller.
    Child classes only need to implement the __init__() and forward() method.
    """

    def __init__(self, options, config, verbose=False):
        super().__init__()

        # read config
        if config is not None:
            self.config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
        else:
            # Since some old checkpoints contained pickled scheduler which needs 'optimizers'
            # module which is now moved into s3prl package.
            original_optimizer = sys.modules.get("optimizers")
            sys.modules["optimizers"] = s3prl.optimizers

            self.all_states = torch.load(options["ckpt_file"], map_location="cpu")
            self.config = self.all_states["Config"]

            del sys.modules["optimizers"]
            if original_optimizer is not None:
                sys.modules["optimizers"] = original_optimizer

        # parse the options dict
        self.load = bool(strtobool(options["load_pretrain"]))
        self.no_grad = bool(strtobool(options["no_grad"]))
        self.permute_input = bool(strtobool(options["permute_input"]))

        # Set model config
        self.model_config = DistillerConfig(self.config["distiller"])
        self.hidden_size = self.model_config.encoder_embed_dim
        self.max_input_length = 0

        if self.max_input_length > 0 and verbose:
            print("[DistillerBuilder] - Maximum input length: ", self.max_input_length)

    def load_model(self, model, state_dict, verbose=False):
        try:
            model.load_state_dict(state_dict)
            if verbose:
                print("[DistillerBuilder] - Pre-trained weights loaded!")
            return model
        except:
            raise RuntimeError("[DistillerBuilder] - Pre-trained weights NOT loaded!")

    def process_input_data(self, wave, wave_len):
        """Process input data for the model"""

        # add arbitary batch axis B if input `wave` has shape of T
        if wave.dim() == 1:
            wave = wave.unsqueeze(0)
        elif wave.dim() > 2:
            raise ValueError

        batch_size = wave.shape[0]
        seq_len = wave.shape[1]

        pad_mask = np.ones((batch_size, seq_len))  # (batch_size, seq_len)

        # zero vectors for padding dimension
        for idx in range(wave.shape[0]):
            pad_mask[idx, wave_len[idx] :] = 0

        wave = wave.to(dtype=torch.float32)  # (batch_size, seq_len, 1)
        pad_mask = torch.FloatTensor(pad_mask).to(
            device=wave.device, dtype=torch.float32
        )  # (batch_size, seq_len)
        return wave, pad_mask  # (x, pad_mask)

    def _forward(self, x, x_len, get_hidden=False, no_pred=False):
        wave, pad_mask = self.process_input_data(x, x_len)
        x = self.model(wave, pad_mask, get_hidden=get_hidden, no_pred=no_pred)

        # x: (feat, feat_final, pred, pad_mask)
        return x


class PretrainedDistiller(DistillerBuilder):
    """
    Use this class to extract features from the Distiller model,
    or to finetune the pre-trained Distiller with any downstream tasks.
    """

    def __init__(self, options, config=None, verbose=False):
        super().__init__(options, config, verbose)

        # Build model
        self.model = DistillerModel(self.model_config)
        self.model.eval() if self.no_grad else self.model.train()
        self.out_dim = self.hidden_size

        # Load from a PyTorch state_dict
        if self.load:
            self.model = self.load_model(
                self.model, self.all_states["Distiller"], verbose
            )
            if verbose:
                print(
                    "[PretrainedDistiller] - Number of parameters: "
                    + str(
                        sum(
                            p.numel()
                            for p in self.model.parameters()
                            if p.requires_grad
                        )
                    )
                )

    def forward(self, wave_inputs, get_hidden=False, no_pred=False):
        wave_len = [len(wave) for wave in wave_inputs]
        wave_inputs = pad_sequence(wave_inputs, batch_first=True)
        # (batch_size, audio_len)

        if self.no_grad:
            with torch.no_grad():
                x = self._forward(
                    wave_inputs, wave_len, get_hidden=get_hidden, no_pred=no_pred
                )
        else:
            x = self._forward(
                wave_inputs, wave_len, get_hidden=get_hidden, no_pred=no_pred
            )
        return x
