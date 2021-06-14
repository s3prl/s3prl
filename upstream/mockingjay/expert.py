# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/mockingjay/expert.py ]
#   Synopsis     [ the mockingjay wrapper ]
#   Author       [ Andy T. Liu (https://github.com/andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


from typing import List, Tuple
from collections import OrderedDict

import yaml
import torch
from torch.tensor import Tensor

from upstream.interfaces import UpstreamBase
from .builder import PretrainedTransformer


class UpstreamExpert(UpstreamBase):
    """
    The Mockingjay wrapper
    """

    def __init__(self, ckpt, model_config=None, **kwargs):
        super().__init__(**kwargs)

        if model_config is not None:
            print(
                "[UpstreamExpert] - Using upstream expert config file from:",
                model_config,
            )
            with open(model_config, "r") as file:
                options = yaml.load(file, Loader=yaml.FullLoader)
        else:
            print("[UpstreamExpert] - Using the default upstream expert config")
            options = {
                "load_pretrain": "True",
                "no_grad": "False",
                "dropout": "default",
                "spec_aug": "False",
                "spec_aug_prev": "True",
                "weighted_sum": "False",
                "permute_input": "False",
            }

        options["ckpt_file"] = ckpt
        options["select_layer"] = -1

        self.transformer = PretrainedTransformer(options, inp_dim=-1)
        assert hasattr(
            self.transformer, "extracter"
        ), "This wrapper only supports `on-the-fly` ckpt with built in feature extracters."

        if len(self.hooks) == 0:
            encoder_path = "self.transformer.model.encoder"

            for i in range(len(self.transformer.model.encoder.layer)):
                self.add_hook(f"{encoder_path}.layer[{i}]", lambda i, o: i[0])
            self.add_hook(
                encoder_path, lambda i, o: o[1][0] if isinstance(o, tuple) else o[0]
            )

            def hook_postprocess(
                hiddens: List[Tuple[str, Tensor]]
            ) -> List[Tuple[str, Tensor]]:
                updated_hiddens_dict = OrderedDict()
                for identifier, tensor in hiddens:
                    if not identifier in updated_hiddens_dict:
                        updated_hiddens_dict[identifier] = [tensor]
                    else:
                        updated_hiddens_dict[identifier].append(tensor)

                updated_hiddens = []
                for identifier, tensors in updated_hiddens_dict.items():
                    updated_hiddens.append((identifier, torch.cat(tensors, dim=1)))

                return updated_hiddens

            self.hook_postprocess = hook_postprocess

    def forward(self, wavs):
        features = self.transformer(wavs)  # (batch_size, extracted_seqlen, feature_dim)
        return {"default": features}
